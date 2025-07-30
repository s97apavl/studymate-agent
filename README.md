# StudyMate Agent

**StudyMate** is an AI-powered research assistant that reads, summarizes, translates, and answers questions about academic PDFs using Mistral LLM and custom tool calls. It also finds similar papers via Google Scholar through SerpAPI.

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourname/studymate-agent.git
cd studymate-agent
```

2. **Create a virtual environment (optional but recommended)**

```bash
python -m venv studymate_venv
source studymate_venv/bin/activate  # or .\studymate_venv\Scripts\activate on Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up API keys**

Create a `.env` file in the root folder with the following variables:

```env
MISTRAL_API_KEY=your-mistral-api-key
HUGGINGFACE_TOKEN=your-huggingface-token
GOOGLE_SCHOLAR_SEARCH=your-serpapi-key
```

---

## 🚀 Running StudyMate

### 🔧 Prompt mode (CLI)

To run a one-time prompt in the console:

```bash
python main.py --mode prompt --prompt "Translate the abstract of the PDF pdfs/paper.pdf into Spanish"
```

This will execute the full pipeline and print both the final conversation and final answer.

### 💬 Gradio UI mode

To start the interactive web interface:

```bash
python main.py --mode ui
```

Then open the URL shown in the terminal. You can upload a PDF and chat with the assistant.

---

## 🗂️ Project Structure

```
studymate-agent/
├── main.py                     # CLI launcher (entry point)
├── pdfs/                       # Folder for user-uploaded PDFs
├── .env                        # API keys (not committed)
├── .gitignore
├── README.md
├── requirements.txt
├── run_script.sh               # Optional bash wrapper
└── studymate_agent/            # Core logic
    ├── __init__.py
    ├── agent_loop.py           # Mistral-based multi-step tool-calling loop
    ├── run_prompt.py           # CLI mode logic
    ├── tools.py                # Tool call implementations
    ├── ui.py                   # Gradio interface
    └── utils.py                # Utilities (PDF handling, model routing, retry, etc.)
```

---

## ✅ Features

- 📄 PDF summarization, Q&A, translation
- 🌍 Multilingual translation via Google Translate
- 🔎 Similar paper search using SerpAPI + Google Scholar
- 💡 Open-ended prompting in CLI mode
- 💬 Memory-aware chatbot in UI (retains latest 2 user-assistant turns)

---

## 🔐 Notes

- `.env` must be created manually and **should never be committed** to version control.
- The PDF you upload will be automatically renamed to `pdfs/uploaded_paper.pdf` and used in all prompts.
- Hugging Face token is used to download the summarization model (`facebook/bart-large-cnn`).
- Mistral API is used for the main assistant logic with tool-calling.

---

Happy researching! 🚀
