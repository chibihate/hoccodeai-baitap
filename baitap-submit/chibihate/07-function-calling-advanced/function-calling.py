import json
from openai import OpenAI
import os
from dotenv import load_dotenv
import inspect
from pydantic import TypeAdapter
import requests
import gradio as gr

# Load environment variables from .env file
load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)
COMPLETION_MODEL = "openai/gpt-oss-20b"
JINA_KEY=os.getenv("JINA_KEY")

def view_website(url: str):
    """
    Summarize the website by JINA
    :param url: URL of website.
    :output: The content of summarization
    """
    url = f'https://r.jina.ai/{url}'
    headers = {
        'Authorization': "Bearer " + JINA_KEY
    }
    response = requests.get(url, headers=headers)
    return response.text

view_website_function = {
    "name": "view_website",
    "description": inspect.getdoc(view_website),
    "parameters": TypeAdapter(view_website).json_schema(),
}

tools = [
    {
        "type": "function",
        "function": view_website_function
    }
]

def summarize(url: str) -> str:
    messages = [{"role": "user", "content": "Tóm tắt nội dung từ " + url}]
    response = client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=messages,
        tools=tools
    )

    tool_call = response.choices[0].message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)

    if tool_call.function.name == 'view_website':
        content_result = view_website(arguments.get('url'))
        messages.append(response.choices[0].message)
        messages.append({
            "role": "tool",
            "content": content_result,
            "tool_call_id": tool_call.id
        })

    return messages

def chat_logic(message, chat_history):
    messages = summarize(message)
    chat_history.append([message, "Waiting..."])
    yield "", chat_history
    response = client.chat.completions.create(
        messages = messages,
        model=COMPLETION_MODEL,
        stream=True,
    )
    chat_history[-1][1] = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content or ""
        chat_history[-1][1] += delta
        yield "", chat_history
    return "", chat_history

with gr.Blocks() as demo:
    gr.Markdown("## Summarize a website")
    message = gr.Textbox(label="Url")
    chat_bot = gr.Chatbot(label="Summarization")
    message.submit(chat_logic, inputs=[message, chat_bot], outputs=[message, chat_bot])

demo.launch()