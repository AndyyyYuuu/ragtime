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


demo = gr.ChatInterface(fn=chat, examples=["Find a crescendo in Holst's The Planets.", "Describe "], title="Music Score Retrieval")
demo.launch()
