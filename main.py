from pathlib import Path

from document_processor.file_handler import DocumentProcessor
from rag_pipeline.rag import build_rag

if __name__ == "__main__":
    # 1. Обработка документов
    processor = DocumentProcessor()
    files = [open(p, "rb") for p in Path("data").iterdir()]
    docs = processor.process(files)

    # 2. Построение RAG
    qa_chain = build_rag(docs)

    # 3. Диалог
    while True:
        query = input("\nВопрос: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = qa_chain.invoke({"query": query})["result"]
        print("Ответ:", answer)
