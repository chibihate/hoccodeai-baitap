import os
from openai import OpenAI
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

# Load environment variables from .env file
load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

def get_text_from_html(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        content_div = soup.find(id='main-detail')  
        
        if content_div:
            return content_div.get_text(separator='\n', strip=True)
        else:
            print("⚠️ Không tìm thấy nội dung chính.")
            return None
    except Exception as e:
        print(f"❌ Error parsing HTML: {e}")
        return None

def summarize_content(content, url):
    prompt = f"""Tóm tắt ngắn gọn và dễ hiểu nội dung của bài báo dưới đây. Giữ lại thông tin quan trọng, rõ ràng.
        URL: {url}
        Nội dung:
        \"\"\"
        {content}
        \"\"\"
        """
    messages = [
        {"role": "system", "content": "Bạn là một trợ lý tóm tắt văn bản tiếng Việt.",}
    ]
    messages.append({
                "role": "user",
                "content": prompt
    })
    stream = client.chat.completions.create(
        messages = messages,
        model="gemma2-9b-it",
        stream=True
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content != None:
            print(content, end="", flush=True)

print("🔗 Dán link bài báo (hoặc nhập 'exit' để thoát):")
while True:
    url = input("URL: ").strip()
    if url.lower() in ['exit', 'quit']:
        print("👋 Tạm biệt!")
        break

    content = get_text_from_html(url)    
    if not content:
        print("⚠️ Không lấy được nội dung từ trang web.")
        continue
    
    print("⏳ Đang tóm tắt nội dung...\n")
    summary = summarize_content(content, url)