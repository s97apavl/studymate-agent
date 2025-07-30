import json
from utils import safe_complete_call
from mistralai import Mistral

def agent_loop(conversation, tool_schemas, max_rounds=5, names_to_functions=None, client=None, model=None):
    """
    tool_registry: dict[str, Callable] — mapping of tool name to function
    tool_schemas: list[dict] — list of Mistral tool definitions
    conversation: list[dict] — standard chat messages
    """
    result = None
    seen_calls = set()
    conversation = conversation.copy()

    for step in range(max_rounds):
        print(f"\n🧠 Step {step + 1} — calling model...\n")

        if step > 0:
            print(conversation)
            response = safe_complete_call(
                client=client,
                model = model, 
                messages = conversation,
                tool_choice = "auto",
                parallel_tool_calls = False,
            )
        else:
            response = safe_complete_call(
                client=client,
                model = model,
                messages = conversation,
                tools = tool_schemas,
                tool_choice = "any",
                parallel_tool_calls = False,
            )

        # response = run_model(conversation, tool_registry)
        conversation.append(response.choices[0].message)

        print(100 * "▼")
        print(f"Model response:\n{response}")
        print(100 * "▲")

        choice = response.choices[0]
        msg = choice.message

        if choice.finish_reason == "stop":
            print("✅ Model signaled stop — final assistant reply.")
            break

        tool_call = response.choices[0].message.tool_calls[0]
        function_name = tool_call.function.name
        function_params = json.loads(tool_call.function.arguments)
        print("\nfunction_name: ", function_name, "\nfunction_params: ", function_params)


    
        for tool_call in msg.tool_calls:
            function_name = tool_call.function.name
            function_params = json.loads(tool_call.function.arguments)
            
            # Fix incorrect arguments from model
            if function_name == "search_similar":
                if "query" in function_params:
                    function_params["abstract"] = function_params.pop("query")

            call_signature = (function_name, tuple(sorted(function_params.items())))

            if call_signature in seen_calls:
                print(f"🛑 Repeated call to {function_name} with same arguments. Stopping to avoid infinite loop.")
                return conversation
            seen_calls.add(call_signature)

            function_result = names_to_functions[function_name](**function_params)

            print(100 * "▼")
            print(f"Tool Call: {function_name}({function_params})")
            print(f"Function result:\n{function_result}")
            print(100 * "▲")


            conversation.append({
                "role":"tool", 
                "name":function_name, 
                "content":function_result, 
                "tool_call_id":tool_call.id
            })

            
    return conversation