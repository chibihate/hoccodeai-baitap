import os
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

# Load environment variables from .env file
load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

MODEL = "gemma2-9b-it"
TOKEN_LIMIT = 3500
INPUT_FILE = "input.txt"
OUTPUT_FILE = "translated.txt"
TRANSLATION_STYLE_PROMPT = """You are a professional translator. Translate the following text into Vietnamese. 
Keep the tone formal, accurate, and natural, and preserve any formatting or line breaks if applicable.
Only output the translated text.
"""

def count_tokens(text, model=MODEL):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def split_text(text, max_tokens):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        chunk_text = " ".join(current_chunk)
        if count_tokens(chunk_text) >= max_tokens:
            # remove last word and save chunk
            current_chunk.pop()
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def translate_chunk(chunk):
    try:
        messages = [
            {"role": "system", "content": TRANSLATION_STYLE_PROMPT},
            {"role": "user", "content": chunk}
        ]

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ Error translating chunk: {e}")
        return ""

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    content = f.read()

chunks = split_text(content, TOKEN_LIMIT)
translated_chunks = []

for i, chunk in enumerate(chunks, 1):
    print(f"🔄 Đang dịch đoạn {i}/{len(chunks)}...")
    translated = translate_chunk(chunk)
    translated_chunks.append(translated)

full_translation = "\n\n".join(translated_chunks)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(full_translation)
