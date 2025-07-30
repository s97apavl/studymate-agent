import argparse
import subprocess
import sys
import os
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath("studymate_agent"))

from run_prompt import run_prompt
from utils import get_final_answer

from mistralai import Mistral
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, set_seed
import torch
from huggingface_hub import login
import os
from dotenv import load_dotenv
import argparse

def main():
    load_dotenv(".env")
    api_key = os.getenv("MISTRAL_API_KEY")
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    api_search_papers = os.getenv("GOOGLE_SCHOLAR_SEARCH")

    login(token=hf_token)

    model = "mistral-large-latest"
    client = Mistral(api_key=api_key)

    seed = 42
    set_seed(seed)

    model_name = "facebook/bart-large-cnn"
    device = 0 if torch.cuda.is_available() else -1
    tokenizer_bart = AutoTokenizer.from_pretrained(model_name)
    model_bart = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    textgen = pipeline(
        "text-generation",
        model=model_bart,
        tokenizer=tokenizer_bart,
        device=device
    )

    def run_classic_prompt(prompt: str) -> str:
        output = textgen(prompt, max_new_tokens=256, do_sample=False)
        return output[0]["generated_text"].strip()


    parser = argparse.ArgumentParser(description="StudyMate CLI")
    parser.add_argument(
        "--mode",
        choices=["prompt", "ui"],
        required=True,
        help="Mode to run: 'prompt' to run a single prompt, 'ui' to launch Gradio interface."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="User prompt to pass to the agent if using --mode prompt.",
        required=False
    )
    args = parser.parse_args()

    if args.mode == "prompt":
        system_message = {
            "role": "system",
            "content": "Only call functions using their defined parameters. For example, use `abstract`, not `query`, for search_similar."
        }

        user_prompt = args.prompt
        conversation = [system_message, {"role": "user", "content": user_prompt}]

        final_conversation = run_prompt(conversation, run_classic_prompt, client, model)
        print("\\n🧾 Final conversation:\\n")
        print(final_conversation)
        print("\\n🧾 Final Answer:\\n")
        print(get_final_answer(final_conversation))

    elif args.mode == "ui":
        subprocess.run([sys.executable, "studymate_agent/ui.py"])

if __name__ == "__main__":
    
    main()