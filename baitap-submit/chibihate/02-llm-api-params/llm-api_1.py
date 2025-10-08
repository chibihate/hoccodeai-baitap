import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

messages = [
    {"role": "system", "content": "You are a helpful assistant.",}
]

user_input = input("You: ")
messages.append({
        "role": "user",
        "content": user_input
})

stream = client.chat.completions.create(
    messages = messages,
    model="gemma2-9b-it",
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")