# 2. Thay vì hardcode `doc = wiki.page('Hayao_Miyazaki').text`, sử dụng function calling để:
#   - Lấy thông tin cần tìm từ câu hỏi
#   - Dùng `wiki.page` để lấy thông tin về
#   - Sử dụng RAG để có kết quả trả lời đúng.

from wikipediaapi import Wikipedia
from openai import OpenAI
import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import unicodedata
import re
import inspect
from pydantic import TypeAdapter
import json

load_dotenv()

clientChromadb = chromadb.PersistentClient(path="./data")
embedding_function = embedding_functions.DefaultEmbeddingFunction()

def convert_name_to_variable_name(name: str) -> str:
    # Normalize and remove accents
    normalized = unicodedata.normalize("NFKD", name)
    ascii_text = normalized.encode("ascii", "ignore").decode("utf-8")

    # Optionally make it safe as a variable name
    variable_name = re.sub(r'\W|^(?=\d)', '_', ascii_text).lower()
    return variable_name

def is_collection_existed(collection_name: str) -> bool:
    collections = clientChromadb.list_collections()
    for item in collections:
        if item.name.lower() == collection_name.lower():
            return True
    return False

def prepare_data(famous_person_name: str) -> bool:
    collection_name = convert_name_to_variable_name(famous_person_name)
    if is_collection_existed(collection_name) == False:
        wiki = Wikipedia('chibihate', 'en')
        doc = wiki.page(famous_person_name).text
        try:
            collection = clientChromadb.create_collection(name=collection_name,
                                    embedding_function=embedding_function)
            paragraphs = doc.split('\n\n')
            for index, paragraph in enumerate(paragraphs):
                collection.add(documents=[paragraph], ids=[str(index)])
            return True
        except ValueError:
            print(f"ERROR\n")
            return False
    return True

def get_famous_person_info(famous_person_name: str, query: str) -> str:
    """
    Get detail information of the famous person you want to know from Wikipedia data.
    param famous_person_name: The name of famous person
    param query: The question from user
    output: A final information based on the question
    """
    collection_name = convert_name_to_variable_name(famous_person_name)
    is_prepared = prepare_data(famous_person_name)
    if is_prepared == True:
        collection = clientChromadb.get_collection(collection_name)
        q = collection.query(query_texts=[query], n_results=3)
        CONTEXT = q["documents"][0]

        prompt = f"""
        Use the following CONTEXT to answer the QUESTION at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use an unbiased and journalistic tone.

        CONTEXT: {CONTEXT}

        QUESTION: {query}
        """
        return prompt
    return f"""No databased to query"""

FUNCTION_MAP = {
    "get_famous_person_info": get_famous_person_info
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_famous_person_info",
            "description": inspect.getdoc(get_famous_person_info),
            "parameters": TypeAdapter(get_famous_person_info).json_schema(),
        },
    }
]

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

def get_completion(messages):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        tools=tools,
        temperature=0
    )
    return response

clientChromadb.heartbeat()

messages = [
    {"role": "system", "content": "You are a Wikipedia assistant. Use the supplied tools to assist the user. You're analytical and funny guys."},
]

while True:
    query = input("Bạn muốn biết thông tin gì về người nổi tiếng nào?")

    messages.append(
        {"role": "user", "content": query}
    )

    response = get_completion(messages)
    first_choice = response.choices[0]
    finish_reason = first_choice.finish_reason

    while finish_reason != "stop":
        tool_call = first_choice.message.tool_calls[0]

        tool_call_function = tool_call.function
        tool_call_arguments = json.loads(tool_call_function.arguments)

        tool_function = FUNCTION_MAP[tool_call_function.name]
        result = tool_function(**tool_call_arguments)

        messages.append(first_choice.message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call_function.name,
            "content": json.dumps(result)
        })

        response = get_completion(messages)
        first_choice = response.choices[0]
        finish_reason = first_choice.finish_reason

    print(f"Bot answer: {first_choice.message.content}")
    messages.append(
        {"role": "assistant"
        , "content": first_choice.message.content}
    )