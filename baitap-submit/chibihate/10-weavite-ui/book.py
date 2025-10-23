# Viết code để tìm kiếm sách/query từ Weavite
import gradio as gr
import weaviate
from weaviate.embedded import EmbeddedOptions

embedded_options = EmbeddedOptions(
    additional_env_vars={
        "ENABLE_MODULES": "backup-filesystem,text2vec-transformers",
        "BACKUP_FILESYSTEM_PATH": "/tmp/backups",  # Chỉ định thư mục backup
        "LOG_LEVEL": "panic",  # Chỉ định level log, chỉ log khi có lỗi
        "TRANSFORMERS_INFERENCE_API": "http://localhost:8000"  # API của model embedding
    },
    persistence_data_path="data",  # Thư mục lưu dữ liệu
)

vector_db_client = weaviate.WeaviateClient(
    embedded_options=embedded_options
)
vector_db_client.connect()
print("DB is ready: {}".format(vector_db_client.is_ready()))

COLLECTION_NAME = "BookCollection"

def search_book(query):
    book_collection = vector_db_client.collections.get(COLLECTION_NAME)
    response = book_collection.query.near_text(
        query=query, limit=10
    )

    results = []
    for book in response.objects:
        book_tuple = (book.properties['title'], book.properties['author'],
                      book.properties['genre'], book.properties['description'],
                      book.properties['grade'], book.properties['date'])
        results.append(book_tuple)
    return results

HEADERS = [
    "title", "author", "genre", "description", "grade", "date"
]

with gr.Blocks(title="Tìm kiếm sách với Vector Database") as interface:
    query = gr.Textbox(label="Tìm kiếm sách", placeholder="Tên, tác giả, thể loại,...")
    search = gr.Button(value="Search")
    results = gr.Dataframe(headers=HEADERS, label="DANH SÁCH KẾT QUẢ")

    search.click(fn=search_book, inputs=query, outputs=results)

interface.queue().launch()

# Đóng kết nối
vector_db_client.close()