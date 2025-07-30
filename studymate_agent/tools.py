import re
import json
from pathlib import Path
from deep_translator import GoogleTranslator
from llama_index.readers.file import PyMuPDFReader
from utils import load_pdf_chunks, split_text
from serpapi import GoogleSearch
from transformers import pipeline

def extract_text_from_pdf(file_path: str) -> list:
    if not Path(file_path).exists():
        return ["File not found."]
    reader = PyMuPDFReader()
    docs = reader.load(file_path=file_path)
    return "\n".join([doc.text for doc in docs])

langs = {'afrikaans': 'af', 'albanian': 'sq', 'amharic': 'am', 'arabic': 'ar', 'armenian': 'hy', 'assamese': 'as', 'aymara': 'ay', 'azerbaijani': 'az', 'bambara': 'bm', 'basque': 'eu', 'belarusian': 'be', 'bengali': 'bn', 'bhojpuri': 'bho', 'bosnian': 'bs', 'bulgarian': 'bg', 'catalan': 'ca', 'cebuano': 'ceb', 'chichewa': 'ny', 'chinese (simplified)': 'zh-CN', 'chinese (traditional)': 'zh-TW', 'corsican': 'co', 'croatian': 'hr', 'czech': 'cs', 'danish': 'da', 'dhivehi': 'dv', 'dogri': 'doi', 'dutch': 'nl', 'english': 'en', 'esperanto': 'eo', 'estonian': 'et', 'ewe': 'ee', 'filipino': 'tl', 'finnish': 'fi', 'french': 'fr', 'frisian': 'fy', 'galician': 'gl', 'georgian': 'ka', 'german': 'de', 'greek': 'el', 'guarani': 'gn', 'gujarati': 'gu', 'haitian creole': 'ht', 'hausa': 'ha', 'hawaiian': 'haw', 'hebrew': 'iw', 'hindi': 'hi', 'hmong': 'hmn', 'hungarian': 'hu', 'icelandic': 'is', 'igbo': 'ig', 'ilocano': 'ilo', 'indonesian': 'id', 'irish': 'ga', 'italian': 'it', 'japanese': 'ja', 'javanese': 'jw', 'kannada': 'kn', 'kazakh': 'kk', 'khmer': 'km', 'kinyarwanda': 'rw', 'konkani': 'gom', 'korean': 'ko', 'krio': 'kri', 'kurdish (kurmanji)': 'ku', 'kurdish (sorani)': 'ckb', 'kyrgyz': 'ky', 'lao': 'lo', 'latin': 'la', 'latvian': 'lv', 'lingala': 'ln', 'lithuanian': 'lt', 'luganda': 'lg', 'luxembourgish': 'lb', 'macedonian': 'mk', 'maithili': 'mai', 'malagasy': 'mg', 'malay': 'ms', 'malayalam': 'ml', 'maltese': 'mt', 'maori': 'mi', 'marathi': 'mr', 'meiteilon (manipuri)': 'mni-Mtei', 'mizo': 'lus', 'mongolian': 'mn', 'myanmar': 'my', 'nepali': 'ne', 'norwegian': 'no', 'odia (oriya)': 'or', 'oromo': 'om', 'pashto': 'ps', 'persian': 'fa', 'polish': 'pl', 'portuguese': 'pt', 'punjabi': 'pa', 'quechua': 'qu', 'romanian': 'ro', 'russian': 'ru', 'samoan': 'sm', 'sanskrit': 'sa', 'scots gaelic': 'gd', 'sepedi': 'nso', 'serbian': 'sr', 'sesotho': 'st', 'shona': 'sn', 'sindhi': 'sd', 'sinhala': 'si', 'slovak': 'sk', 'slovenian': 'sl', 'somali': 'so', 'spanish': 'es', 'sundanese': 'su', 'swahili': 'sw', 'swedish': 'sv', 'tajik': 'tg', 'tamil': 'ta', 'tatar': 'tt', 'telugu': 'te', 'thai': 'th', 'tigrinya': 'ti', 'tsonga': 'ts', 'turkish': 'tr', 'turkmen': 'tk', 'twi': 'ak', 'ukrainian': 'uk', 'urdu': 'ur', 'uyghur': 'ug', 'uzbek': 'uz', 'vietnamese': 'vi', 'welsh': 'cy', 'xhosa': 'xh', 'yiddish': 'yi', 'yoruba': 'yo', 'zulu': 'zu'}
def translate_text(text: str, language: str) -> str:
    language = language.lower()
    
    language = langs.get(language, language)
    return GoogleTranslator(target=language).translate(text)

def translate_pdf(filepath: str, language: str) -> str:
    language = language.lower()
    language = translate_text("test", language).split()[-1]
    reader = PyMuPDFReader()
    docs = reader.load(file_path=filepath)
    full_text = "\n".join(doc.text for doc in docs)
    chunks = split_text(full_text)
    translated_chunks = [GoogleTranslator(target=language).translate(chunk) for chunk in chunks]
    return "\n\n".join(translated_chunks)

def answering_text(text: str, question: str, run_classic_prompt=None) -> str:
    context = "\n\n".join(text.split("\n\n")[:5])
    prompt = f"""You are a research assistant AI. Based on the context below, provide a helpful and clear answer.

Research Context:
{context}

User Question:
{question}

Answer:"""
    return run_classic_prompt(prompt)

def answering_pdf(filepath: str, question: str, run_classic_prompt=None) -> str:
    chunks = load_pdf_chunks(filepath)
    if not chunks or chunks == ["File not found."]:
        return "❌ PDF file not found or empty."
    selected_chunks = chunks[:5]
    context = "\n\n".join(selected_chunks)
    prompt = f"""You are a research assistant AI. Based on the context below, provide a helpful and clear answer.

Research Context:
{context}

User Question:
{question}

Answer:"""
    return run_classic_prompt(prompt)

def abstract_from_text(text: str) -> str:
    clean_text = re.sub(r"-\n", "", text)
    clean_text = re.sub(r"\n", " ", clean_text)
    match = re.search(r"(?i)abstract\s*[:\-]?\s*(.{300,2000}?)\s+(?=[1I]\.\s*Introduction)", clean_text)
    return match.group(1).strip() if match else "⚠️ Abstract section not found."

def abstract_from_pdf(filepath: str) -> str:
    chunks = load_pdf_chunks(filepath)
    if not chunks or chunks == ["File not found."]:
        return "❌ PDF file not found or empty."
    full_text = "\n".join(chunks)
    return abstract_from_text(full_text)

def search_similar(abstract: str, title: str = "") -> str:
    from os import getenv
    query = f"{title} {abstract[:300]}"
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": getenv("GOOGLE_SCHOLAR_SEARCH")
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    articles = [
        {
            "title": item.get("title", "❓ No title"),
            "authors": item.get("publication_info", {}).get("summary", "❓ No author info"),
            "link": item.get("link", "🔗 No link")
        }
        for item in results.get("organic_results", [])[:5]
    ]
    if not articles:
        return "⚠️ No similar articles found."
    return "\n\n".join([f"{i+1}. {a['title']}\n   Authors: {a['authors']}\n   Link: {a['link']}" for i, a in enumerate(articles)])

def summarize_text(text: str) -> str:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
    return summarizer(text, max_length=512, min_length=30, do_sample=False)[0]['summary_text']

def summarize_pdf(pdf_path: str) -> str:
    if not Path(pdf_path).exists():
        return f"❌ File not found: {pdf_path}"
    reader = PyMuPDFReader()
    docs = reader.load(file_path=pdf_path)
    chunks = [doc.text.strip() for doc in docs if len(doc.text.strip()) > 50]
    if not chunks:
        return "❌ No meaningful text found in PDF."
    summaries = [summarize_text(chunk) for chunk in chunks if len(chunk) > 50]
    combined = " ".join(summaries)[:1024]
    return summarize_text(combined)