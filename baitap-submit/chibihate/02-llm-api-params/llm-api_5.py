from openai import OpenAI
import os
import subprocess
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

def ask_chatgpt(prompt):
    print("\n⏳ Đang hỏi bot, vui lòng chờ...\n")
    response = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[
            {"role": "system", "content": "Bạn là một lập trình viên Python giỏi, chỉ trả lời bằng code chạy được và có hàm main. Bỏ ```python ```` trong phần trả lời và nếu chú thích thì hãy thêm # trước đó, không cần ví dụ"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.95
    )
    code = response.choices[0].message.content
    return code

def save_to_file(code, filename="final.py"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"✅ Code đã được lưu vào {filename}")

def run_code(filename="final.py"):
    print("\n🚀 Đang chạy code...\n")
    try:
        result = subprocess.run(["python", filename], capture_output=True, text=True)
        print("💡 Kết quả:")
        print(result.stdout)
        if result.stderr:
            print("⚠️ Lỗi:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ Lỗi khi chạy code: {e}")

def main():
    print("=== ỨNG DỤNG GIẢI BÀI TẬP LẬP TRÌNH VỚI AI ===")
    user_input = input("Nhập câu hỏi lập trình (ví dụ: Viết hàm kiểm tra số nguyên tố):\n> ")

    prompt = f"Viết mã Python để giải bài toán sau:\n{user_input}\nHãy đưa ra code hoàn chỉnh, có thể chạy độc lập."

    code = ask_chatgpt(prompt)
    print("\n--- Đáp án của bot ---\n")
    print(code)

    save_to_file(code)
    run_code()

if __name__ == "__main__":
    main()