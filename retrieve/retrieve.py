import os

import anthropic
import dotenv
import requests

dotenv.load_dotenv()

PIECES = ["planets"]

PIECE_FINEDESC_PATH = "data/raw_xml/holst-the-planets-op-32.finedesc.txt"

SENSO_LM_KEY = os.environ["SENSO_LMTXT_KEY"]
SENSO_FINE_KEY = os.environ["SENSO_FINETXT_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_KEY"]

AGENT_SYSTEM_PROMPT = open("retrieve/system_prompt.txt").read()

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def text_from_assistant_content(content) -> str:
    # join text segments from an assistant message's content (plain str or Anthropic content blocks)
    if isinstance(content, str):
        return content
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        btype = getattr(block, "type", None)
        if btype == "text" and hasattr(block, "text"):
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def search_by_description(query: str, top_k: int = 3) -> list[str]:
    if top_k < 1: 
        raise ValueError("Top k must be greater than 0")
    response = requests.post("https://apiv2.senso.ai/api/v1/org/search/context",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": SENSO_LM_KEY,
        },
        json={
            "query": query,
            "max_results": top_k,
            "require_scoped_ids": False
        }
    )
    return [r["chunk_text"] for r in response.json()["results"]]

def search_by_notes(query: str, top_k: int = 3) -> list[str]:
    if top_k < 1: 
        raise ValueError("Top k must be greater than 0")
    response = requests.post("https://apiv2.senso.ai/api/v1/org/search/context",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": SENSO_FINE_KEY,
        },
        json={
            "query": query,
            "max_results": top_k,
            "require_scoped_ids": False
        }
    )
    return [r["chunk_text"] for r in response.json()["results"]]

def retrieve_by_piece_bar(piece: str, bar: int) -> list[str]:
    if piece not in PIECES:
        raise ValueError(f"Piece must be one of: {PIECES}")
    if bar < 1:
        raise ValueError("Bar number must be greater than 0")
    with open(PIECE_FINEDESC_PATH, encoding="utf-8") as f:
        lines = f.readlines()
    n = len(lines)
    if bar > n:
        raise ValueError(f"Bar {bar} out of range (finedesc has {n} lines)")
    return [lines[bar - 1].rstrip("\n\r")]


tools = [
    {
        "name": "search_by_description",
        "description": "Retrieve by text matching (simple embedding search) with a description of each bar.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A brief textual search term."
                },
                "top_k": {
                    "type": "integer",
                    "description": "The number of results to return. Defaults to 3."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_by_notes",
        "description": "Retrieve by text matching (simple embedding search) with the fine-grained notes of each bar.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A brief textual search term."
                },
                "top_k": {
                    "type": "integer",
                    "description": "The number of results to return. Defaults to 3."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "retrieve_by_bar",
        "description": "Retrieve notes and some other fine-grained musical details by bar number. ",
        "input_schema": {
            "type": "object",
            "properties": {
                "piece": {
                    "type": "string",
                    "description": f"The piece to search for. Options: {PIECES}"
                },
                "bar": {
                    "type": "integer",
                    "description": "The bar number to search for."
                }
            },
            "required": ["piece", "bar"]
        }
    }
]

def call_agent(query: str, history: list[dict] = [], max_turns: int = 10) -> dict:
    messages = list(history) + [{"role": "user", "content": query}]
    turn_count = 0
    retrievals: list[str] = []
    while True:
        if turn_count > max_turns:
            messages.append({"role": "user", "content": "You have exceeded the maximum number of tool calls. Please summarize the results as best you can and end the conversation."})
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            tools=tools,
            messages=messages,
            system=AGENT_SYSTEM_PROMPT
        )
        print(response.content)

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn" or turn_count > max_turns + 1:
            # last condition is to ensure safety in case the agent really goes wrong
            for block in response.content:
                if hasattr(block, "text"):
                    print(block.text)
            break

        tool_results = []
        for block in response.content:
            turn_count += 1
            if block.type == "tool_use":
                tool_name = block.name
                if tool_name == "search_by_description":
                    query = block.input["query"]
                    # default top 3 defined here
                    results = search_by_description(query, block.input.get("top_k", 3))
                elif tool_name == "search_by_notes":
                    query = block.input["query"]
                    results = search_by_notes(query, block.input.get("top_k", 3))
                elif tool_name == "retrieve_by_bar":
                    piece = block.input["piece"]
                    bar = block.input["bar"]
                    results = retrieve_by_piece_bar(piece, bar)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "\n".join(
                            f"Result {i + 1}: {chunk}" for i, chunk in enumerate(results)
                        ),
                    }
                )
                retrievals.extend(results)

        messages.append({"role": "user", "content": tool_results})
    
    last = messages[-1]
    if last.get("role") != "assistant":
        return {"answer": ""}
    return {"answer": text_from_assistant_content(last["content"]), "retrievals": retrievals}

if __name__ == "__main__":
    query = "Find a crescendo in Holst's The Planets."
    results = call_agent(query)
    print(results)