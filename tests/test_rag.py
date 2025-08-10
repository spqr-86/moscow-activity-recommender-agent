from langchain.schema import Document

from rag_pipeline import rag


def test_build_rag_with_docs(monkeypatch):
    docs = [Document(page_content="Тестовый документ")]

    class DummyRetriever:
        def retrieve(self, query):
            return ["ответ"]

    class DummyQAChain:
        def run(self, query):
            return "ответ"

        @classmethod
        def from_chain_type(cls, llm, retriever):
            assert retriever is not None
            return cls()

    # Мокаем RetrievalQA.from_chain_type, чтобы не дергать реальный LLM
    monkeypatch.setattr(
        rag,
        "RetrievalQA",
        type("DummyClass", (), {"from_chain_type": DummyQAChain.from_chain_type}),
    )

    chain = rag.build_rag(docs)
    answer = chain.run("Что это?")
    assert answer == "ответ"
