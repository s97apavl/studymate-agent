import re
import textwrap
import json
import random
import inspect
import functools
import time
from pathlib import Path
from llama_index.readers.file import PyMuPDFReader
from mistralai.models import SDKError


def run_model(conversation, tools, tool_choice="any", parallel_tool_calls=False, client=None, model=None):
    return client.chat.complete(
        model=model,
        messages=conversation,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls
    )

def load_pdf_chunks(file_path: str) -> list:
    if not Path(file_path).exists():
        return ["File not found."]
    reader = PyMuPDFReader()
    docs = reader.load(file_path=file_path)
    return [doc.text for doc in docs]

def split_text(text, max_length=4500):
    return textwrap.wrap(text, max_length)

def extract_tool_calls(decoded: str):
    matches = re.findall(r'\[\s*{.*?}\s*}\s*\]', decoded, re.DOTALL)
    tool_calls = []
    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, list) and "name" in data[0] and "arguments" in data[0]:
                tool_calls.extend(data)
        except Exception:
            continue
    return tool_calls

def generate_tool_call_id():
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=9))

def build_tool_schema(fn):
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or ""
    first_line = doc.splitlines()[0] if doc else "No description provided."

    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

    for name, param in sig.parameters.items():
        parameters["properties"][name] = {
            "type": "string",
            "description": f"{name} parameter"
        }
        if param.default is inspect.Parameter.empty:
            parameters["required"].append(name)

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": first_line.strip(),
            "parameters": parameters
        }
    }

def build_function_map_with_partial(tools, **shared_kwargs):
    return {
        fn.__name__: functools.partial(fn, **shared_kwargs)
        for fn in tools
    }

def safe_complete_call(client, **kwargs):
    retries = 3
    delay = 2
    for attempt in range(retries):
        try:
            return client.chat.complete(**kwargs)
        except SDKError as e:
            if e.status_code == 429:
                print(f"⚠️ Capacity error (429). Retry {attempt + 1}/{retries} in {delay}s...")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("❌ Failed after 3 retries due to service capacity limits.")

def get_final_answer(final_conversation):
    return final_conversation[-1].content