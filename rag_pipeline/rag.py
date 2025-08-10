from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

CHROMA_PERSIST_DIR = Path("chroma_db")
CHROMA_PERSIST_DIR.mkdir(exist_ok=True)


def build_rag(docs):
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        # huggingfacehub_api_token=config.HF_API_TOKEN
    )

    # Проверяем, есть ли база уже на диске
    if any(CHROMA_PERSIST_DIR.iterdir()):
        print("Загрузка существующего хранилища Chroma из", CHROMA_PERSIST_DIR)
        vectorstore = Chroma(
            embedding_function=embeddings, persist_directory=str(CHROMA_PERSIST_DIR)
        )
    else:
        print("Создаём новое хранилище Chroma и сохраняем в", CHROMA_PERSIST_DIR)
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=str(CHROMA_PERSIST_DIR),
        )
        vectorstore.persist()

    retriever = vectorstore.as_retriever()

    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2, max_tokens=500)

    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    return qa_chain
