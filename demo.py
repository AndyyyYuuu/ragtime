import gradio as gr
from retrieve import call_agent


def chat(message, history):
    messages = []
    for m in history or []:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
            text = m.get("content")
            text = text.strip() if isinstance(text, str) else str(text or "").strip()
            if text:
                messages.append({"role": m["role"], "content": text})
    call = call_agent(message, messages, max_turns=3)
    print("\n\nRetrievals:\n", call["retrievals"])
    return call["answer"]

examples = [line.strip().split("#")[0].strip() for line in open("examples.txt").readlines() if line.strip() != ""]

demo = gr.ChatInterface(fn=chat, examples=examples, title="Demo: Retrieval for Holst's The Planets")
demo.launch(share=False)
