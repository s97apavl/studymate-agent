
from utils import (
    build_tool_schema,
    build_function_map_with_partial,
)
from tools import (
    translate_text, translate_pdf, answering_text, answering_pdf,
    abstract_from_text, abstract_from_pdf, search_similar,
    summarize_text, summarize_pdf
)
from agent_loop import agent_loop
from functools import partial, update_wrapper

def run_prompt(conversation, run_classic_prompt, client, model):
    
    anstxt = partial(answering_text, run_classic_prompt=run_classic_prompt)
    update_wrapper(anstxt, answering_text)

    anspdf = partial(answering_pdf, run_classic_prompt=run_classic_prompt)
    update_wrapper(anspdf, answering_pdf)
    tools = [
        translate_text, translate_pdf, anstxt, anspdf, abstract_from_text, abstract_from_pdf, search_similar, summarize_text, summarize_pdf
    ]

    mistral_tool_definitions = [build_tool_schema(fn) for fn in tools]
    names_to_functions = build_function_map_with_partial(tools)

    MAX_TOKENS = 512
    final_conversation = agent_loop(conversation, mistral_tool_definitions, MAX_TOKENS, names_to_functions, client=client, model=model)
    return final_conversation