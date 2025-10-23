# Áp dụng Weaviate để tạo ra một RAG flow đơn giản, gợi ý sách hay, dựa theo query của người dùng.
# Thay vì hard code query, hãy lấy query từ người dùng input ở console, hoặc tạo app bằng Gradio.
import weaviate
from weaviate.embedded import EmbeddedOptions
from weaviate.classes.config import Configure, Property, DataType, Tokenization
import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter
import gradio as gr

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

embedded_options = EmbeddedOptions(
    additional_env_vars={
        # Kích hoạt các module cần thiết, nhớ thêm generative-openai
        "ENABLE_MODULES": "backup-filesystem,text2vec-transformers,generative-openai",
        "BACKUP_FILESYSTEM_PATH": "/tmp/backups",  # Chỉ định thư mục backup
        "LOG_LEVEL": "panic",  # Chỉ định level log, chỉ log khi có lỗi
        "TRANSFORMERS_INFERENCE_API": "http://localhost:8000",  # API của model embedding
        "OPENAI_APIKEY": OPENAI_API_KEY
    },
    persistence_data_path="data",  # Thư mục lưu dữ liệu
)

vector_db_client = weaviate.WeaviateClient(
    embedded_options=embedded_options
)
vector_db_client.connect()
print("DB is ready: {}".format(vector_db_client.is_ready()))

COLLECTION_NAME = "BookCollectionRAG"

def create_collection():
    # Tạo schema cho collection
    book_collection = vector_db_client.collections.create(
        name=COLLECTION_NAME,
        # Sử dụng model transformers để tạo vector
        vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
        generative_config=Configure.Generative.openai(
            model="gpt-4o",
        ),
        properties=[
            Property(name="title", data_type=DataType.TEXT,
                     vectorize_property_name=True, tokenization=Tokenization.LOWERCASE),
            Property(name="author", data_type=DataType.TEXT, tokenization=Tokenization.LOWERCASE),
            Property(name="description", data_type=DataType.TEXT, tokenization=Tokenization.LOWERCASE),
            Property(name="grade", data_type=DataType.NUMBER),
            Property(name="genre", data_type=DataType.TEXT, tokenization=Tokenization.WORD),
            Property(name="lexile", data_type=DataType.NUMBER),
            Property(name="path", data_type=DataType.TEXT, tokenization=Tokenization.WHITESPACE),
            Property(name="is_prose", data_type=DataType.INT),
            Property(name="date", data_type=DataType.TEXT, tokenization=Tokenization.LOWERCASE),
            Property(name="intro", data_type=DataType.TEXT, tokenization=Tokenization.LOWERCASE),
            Property(name="excerpt", data_type=DataType.TEXT, tokenization=Tokenization.LOWERCASE),
            Property(name="license", data_type=DataType.TEXT, tokenization=Tokenization.LOWERCASE),
            Property(name="notes", data_type=DataType.TEXT, tokenization=Tokenization.LOWERCASE)
        ]
    )

    # Set the path to the file you'd like to load
    file_path = "commonlit_texts.csv"

    # Load the latest version
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "kononenko/commonlit-texts",
        file_path,
        )
    df = df.replace({np.nan: None})

    # Chuyển đổi dữ liệu để import
    sent_to_vector_db = df.to_dict(orient='records')
    total_records = len(sent_to_vector_db)
    print(f"Inserting data to Vector DB. Total records: {total_records}")

    # Import dữ liệu vào DB theo batch
    with book_collection.batch.dynamic() as batch:
        for data_row in sent_to_vector_db:
            print(f"Inserting: {data_row['title']}")
            batch.add_object(properties=data_row)

    print("Data saved to Vector DB")

def search_book(query: str):
    if vector_db_client.collections.exists(COLLECTION_NAME):
        print("Collection {} already exists".format(COLLECTION_NAME))
    else:
        create_collection()

    books = vector_db_client.collections.get(COLLECTION_NAME)

    response = books.generate.near_text(
        query=query,
        single_prompt="Viết một bài giới thiệu ngắn gọn tiếng Việt về sách: {title}, tác giả: {author}, tóm tắt: {description}.",
        limit=5
    )
    results = []
    for book in response.objects:
        book_summary = (book.properties['title'], book.properties['author'], book.generative.text)
        results.append(book_summary)
    return results

HEADERS = [
    "title", "author", "description"
]

with gr.Blocks(title="Tìm kiếm sách với Vector Database") as interface:
    query = gr.Textbox(label="Tìm kiếm sách với bài giới thiệu ngắn gọn", placeholder="Tên sách, tác giả, thể loại,...")
    search = gr.Button(value="Search")
    results = gr.Dataframe(headers=HEADERS, label="Danh sách kết quả")

    search.click(fn=search_book, inputs=query, outputs=results)

interface.queue().launch()

# Đóng kết nối
vector_db_client.close()