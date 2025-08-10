import hashlib
import json
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from docling.document_converter import DocumentConverter
from langchain.schema import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from config import constants
from config.settings import settings


class DocumentProcessor:
    def __init__(self):
        self.headers = [("#", "Header 1"), ("##", "Header 2")]
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def validate_files(self, files: List) -> None:
        total_size = sum(os.path.getsize(f.name) for f in files)
        if total_size > constants.MAX_TOTAL_SIZE:
            raise ValueError(
                "Суммарный размер файлов превышает "
                + f"{constants.MAX_TOTAL_SIZE//1024//1024} МБ"
            )

    def process(self, files: List) -> List[Document]:
        self.validate_files(files)
        all_chunks = []
        seen_hashes = set()

        for file in files:
            try:
                with open(file.name, "rb") as f:
                    file_hash = self._generate_hash(f.read())

                cache_path = self.cache_dir / f"{file_hash}.pkl"

                if self._is_cache_valid(cache_path):
                    chunks = self._load_from_cache(cache_path)
                else:
                    chunks = self._process_file(file)
                    self._save_to_cache(chunks, cache_path)

                for chunk in chunks:
                    chunk_hash = self._generate_hash(chunk.page_content.encode())
                    if chunk_hash not in seen_hashes:
                        all_chunks.append(chunk)
                        seen_hashes.add(chunk_hash)

            except Exception as e:
                print(f"Ошибка при обработке {file.name}: {e}")
                continue

        return all_chunks

    def _process_file(self, file) -> List[Document]:
        ext = Path(file.name).suffix.lower()

        if ext in [".pdf", ".docx", ".txt", ".md"]:
            return self._process_doc(file)
        elif ext == ".json":
            return self._process_json(file)
        else:
            print(f"Пропуск неподдерживаемого типа файла: {file.name}")
            return []

    def _process_doc(self, file) -> List[Document]:
        converter = DocumentConverter()
        markdown = converter.convert(file.name).document.export_to_markdown()
        splitter = MarkdownHeaderTextSplitter(self.headers)
        return splitter.split_text(markdown)

    def _process_json(self, file) -> List[Document]:
        try:
            with open(file.name, "r", encoding="utf-8") as f:
                data = json.load(f)

            texts = self._extract_text_from_json(data)
            docs = [Document(page_content=text) for text in texts if text.strip()]
            return docs

        except Exception as e:
            print(f"Ошибка при обработке JSON {file.name}: {e}")
            return []

    def _extract_text_from_json(self, data):
        """Рекурсивно ищет все строки в JSON."""
        texts = []
        if isinstance(data, dict):
            for v in data.values():
                texts.extend(self._extract_text_from_json(v))
        elif isinstance(data, list):
            for item in data:
                texts.extend(self._extract_text_from_json(item))
        elif isinstance(data, str):
            texts.append(data)
        return texts

    def _generate_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _save_to_cache(self, chunks: List[Document], cache_path: Path):
        with open(cache_path, "wb") as f:
            pickle.dump({"timestamp": datetime.now().timestamp(), "chunks": chunks}, f)

    def _load_from_cache(self, cache_path: Path) -> List[Document]:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["chunks"]

    def _is_cache_valid(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age < timedelta(days=settings.CACHE_EXPIRE_DAYS)
