import pytest
from scripts.rag_setup import retrieve

def test_retrieve():
    results = retrieve("где провести впемя семьей в Москве")
    assert len(results) > 0, "Нет результатов"
    assert any("музей" in doc.page_content.lower() for doc in results), "Не релевантно"
