import json
import os
import logging
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Модель эмбеддингов (понимает русский)
embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="intfloat/multilingual-e5-large",
    task="feature-extraction",
    huggingfacehub_api_token=hf_token
)

# ===== Загрузка данных =====
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    documents = []
    for item in data:
        content = f"{item.get('title', '')} - {item.get('description', '')} " \
                  f"(Категория: {item.get('categories', 'N/A')}, " \
                  f"Адрес: {item.get('address', 'N/A')})"
        doc = Document(
            page_content=content,
            metadata={
                'id': str(item.get('id', '')),
                'type': 'place' if 'places' in file_path else 'event'
            }
        )
        documents.append(doc)
    logger.info(f"Loaded {len(documents)} documents from {file_path}")
    return documents

# ===== Инициализация векторного хранилища =====
def init_vectorstore():
    places_docs = load_data('data/places.json')
    events_docs = load_data('data/events.json')
    all_docs = places_docs + events_docs

    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection_name="moscow_activities",
        persist_directory="./chroma_db"
    )
    logger.info(f"RAG база создана: {len(all_docs)} документов")
    return vectorstore

# ===== Получение ретривера =====
def get_retriever():
    if not os.path.exists("./chroma_db") or not os.listdir("./chroma_db"):
        logger.info("Векторная база не найдена — создаём заново...")
        init_vectorstore()

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# ===== Основной поиск =====
def retrieve(query):
    logger.info(f"Retrieving for query: {query}")
    retriever = get_retriever()
    results = retriever.invoke(query)  # Новый метод вместо get_relevant_documents
    logger.info(f"Retrieved {len(results)} documents")
    return results

# ===== Точка входа =====
if __name__ == "__main__":
    init_vectorstore()
