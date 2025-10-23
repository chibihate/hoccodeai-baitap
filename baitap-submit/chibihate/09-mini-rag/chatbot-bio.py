# 1. Dùng chunking để làm bot trả lời tiểu sử người nổi tiếng, anime v...v
#   - <https://en.wikipedia.org/wiki/S%C6%A1n_T%C3%B9ng_M-TP>
#   - <https://en.wikipedia.org/wiki/Jujutsu_Kaisen>
from wikipediaapi import Wikipedia
from openai import OpenAI
import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

COLLECTION_NAME = "Son_Tung_MTP"
clientChromadb = chromadb.PersistentClient(path="./data")
clientChromadb.heartbeat()

embedding_function = embedding_functions.DefaultEmbeddingFunction()
if clientChromadb.count_collections() == 0:
    collection = clientChromadb.create_collection(name=COLLECTION_NAME,
                                    embedding_function=embedding_function)

    wiki = Wikipedia('chibihate', 'en')
    doc = wiki.page('Sơn_Tùng_M-TP').text
    paragraphs = doc.split('\n\n')

    for index, paragraph in enumerate(paragraphs):
        collection.add(documents=[paragraph], ids=[str(index)])
    
collection = clientChromadb.get_collection(COLLECTION_NAME)

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

while True:
    query = input("Bạn muốn biết thông tin gì về Sơn Tùng?")

    if query.lower() in ["no", "exit", "close"]:
        break

    q = collection.query(query_texts=[query], n_results=3)
    CONTEXT = q["documents"][0]

    prompt = f"""
    Use the following CONTEXT to answer the QUESTION at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use an unbiased and journalistic tone.

    CONTEXT: {CONTEXT}

    QUESTION: {query}
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "user", "content": prompt},
        ]
    )

    print(response.choices[0].message.content)