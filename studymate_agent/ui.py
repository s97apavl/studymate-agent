import gradio as gr
import tempfile
from agent_loop import agent_loop
from tools import (
    translate_text, translate_pdf, answering_text, answering_pdf,
    abstract_from_text, abstract_from_pdf, search_similar,
    summarize_text, summarize_pdf
)
from utils import build_function_map_with_partial, build_tool_schema
import os

# === Shared PDF context and conversation history ===
pdf_context = {
    "filepath": None,
    "history": []
}

system_prompt = "You are StudyMate, an assistant that helps students understand academic papers by summarizing, translating, and answering questions."

# === Register tools and build schemas ===
tools = [
    translate_text, translate_pdf, answering_text, answering_pdf,
    abstract_from_text, abstract_from_pdf, search_similar,
    summarize_text, summarize_pdf
]
mistral_tool_definitions = [build_tool_schema(fn) for fn in tools]
names_to_functions = build_function_map_with_partial(tools)

# === Handle PDF upload and store its temporary path ===
def process_pdf(file):
    """
    Save the uploaded PDF to a known location with readable name,
    and reset conversation context.
    """
    try:
        filename = "uploaded_paper.pdf"
        pdf_dir = os.path.join(os.getcwd(), "pdfs")
        os.makedirs(pdf_dir, exist_ok=True)
        full_path = os.path.join(pdf_dir, filename)

        with open(full_path, "wb") as f:
            f.write(file)

        # Set context
        pdf_context["filepath"] = full_path
        pdf_context["history"] = [{"role": "system", "content": system_prompt}]

        print(f"pdf_context: \n{pdf_context}")
        return "✅ PDF uploaded and ready for questions like 'summarize', 'translate', or 'find similar articles'."

    except Exception as e:
        return f"❌ Error: {str(e)}"


# === Chat interaction handler with persistent conversation ===
def chat_with_agent(history, user_input):
    """
    Process user input, attach PDF path if needed, run agent, and maintain trimmed conversation.
    """
    try:
        # Internally add path to prompt if needed, without exposing it to user
        injected_input = user_input
        if pdf_context["filepath"] and "pdf_path" not in user_input and "pdfs/" not in user_input:
            injected_input += f'\nThe file is called "{pdf_context["filepath"]}".'

        # Limit history to last 2 exchanges
        trimmed_history = pdf_context["history"][-5:]  # include system + 2 rounds (user+assistant)
        conversation = trimmed_history + [{"role": "user", "content": injected_input}]
        MAX_TOKENS = 512

        from mistralai import Mistral
        api_key = os.getenv("MISTRAL_API_KEY")
        model = "mistral-large-latest"
        client = Mistral(api_key=api_key)

        final_conversation = agent_loop(
            conversation, mistral_tool_definitions, MAX_TOKENS, names_to_functions, client=client, model=model
        )
        reply = final_conversation[-1].content

        # Save only the last 2 exchanges in history (excluding system message)
        pdf_context["history"] = (
            [h for h in pdf_context["history"] if h["role"] == "system"] +
            final_conversation[-4:]  # 2 rounds (user+assistant)
        )

    except Exception as e:
        reply = f"❌ Error: {str(e)}"

    # Update visible chat history
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})
    return history, history



# === Gradio UI definition ===
with gr.Blocks() as demo:
    gr.Markdown("# 📘 StudyMate — Chat with Academic PDFs")

    with gr.Row():
        file = gr.File(label="Upload PDF", type="binary", file_types=[".pdf"])
        output = gr.Textbox(label="Status", interactive=False)

    upload_btn = gr.Button("Process PDF")
    chatbot = gr.Chatbot(type="messages")
    state = gr.State([])

    with gr.Row():
        msg = gr.Textbox(show_label=False, placeholder="Ask something like: 'summarize this', 'translate to German', or 'find similar papers'...")
        send = gr.Button("Send")

    upload_btn.click(process_pdf, inputs=file, outputs=output)
    send.click(chat_with_agent, inputs=[state, msg], outputs=[chatbot, state])
    msg.submit(chat_with_agent, inputs=[state, msg], outputs=[chatbot, state])

demo.launch(share=True)