import os
import json
import logging
from typing import Any, Dict, List

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredExcelLoader

from deepsearcher.loader.file_loader.base import BaseLoader

class ExcelRAGLoader(BaseLoader):
    def __init__(
        self,
        rows_per_chunk: int = 36,
        max_empty_rows_between_tables: int = 2,
        max_metadata_bytes: int = 60_000,
    ):
        """
        :param rows_per_chunk: Number of rows per chunk for splitting tabular data.
        :param max_empty_rows_between_tables: Max consecutive empty rows to merge into one table.
        :param max_metadata_bytes: Maximum size of metadata JSON in bytes.
        """
        super().__init__()
        self.rows_per_chunk = rows_per_chunk
        self.max_empty_rows_between_tables = max_empty_rows_between_tables
        self.max_metadata_bytes = max_metadata_bytes

        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

        # Text splitter for non-table content
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=280,
            separators=["\n\n", "\n", ".", "!", "?"]
        )

    def _sanitize_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Truncate or drop metadata fields so that JSON size stays below max_metadata_bytes.
        """
        try:
            raw = json.dumps(meta, ensure_ascii=False).encode("utf-8")
        except Exception:
            return {k: meta[k] for k in ("source", "sheet_name", "content_type") if k in meta}

        if len(raw) <= self.max_metadata_bytes:
            return meta

        # Drop lower-priority keys first
        for key in ("chunk_id", "columns", "first_row", "last_row", "has_header"):
            if key in meta:
                del meta[key]
                raw = json.dumps(meta, ensure_ascii=False).encode("utf-8")
                if len(raw) <= self.max_metadata_bytes:
                    return meta

        # Fallback to essentials only
        return {k: meta[k] for k in ("source", "sheet_name", "content_type") if k in meta}

    def load_file(self, file_path: str) -> List[Document]:
        """
        Load an Excel file, splitting into table- and text-based chunks,
        and ensure metadata stays within size limits.
        """
        documents: List[Document] = []
        try:
            sheets = pd.read_excel(file_path, sheet_name=None, header=None)
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            return []
        except pd.errors.EmptyDataError:
            self.logger.error(f"Empty data in file: {file_path}")
            return []
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            return []

        for sheet_name, df in sheets.items():
            if df.empty:
                self.logger.info(f"Skipping empty sheet: {sheet_name}")
                continue

            tables = self._detect_tables(df)
            if tables:
                # Structured tables
                for t_idx, t_df in enumerate(tables):
                    has_hdr = bool(t_df.attrs.get("header_rows", 0))
                    if has_hdr:
                        raw_hdr = t_df.iloc[0].fillna("Unnamed").astype(str).tolist()
                        headers = self._make_unique_headers(raw_hdr)
                        data = t_df.iloc[1:].reset_index(drop=True)
                    else:
                        headers = [f"Column_{i}" for i in range(t_df.shape[1])]
                        data = t_df.copy()
                    data.columns = headers
                    data = data.fillna("N/A").infer_objects(copy=False)

                    for start in range(0, len(data), self.rows_per_chunk):
                        end = min(start + self.rows_per_chunk, len(data))
                        chunk_df = data.iloc[start:end]
                        first = start + (2 if has_hdr else 1)
                        last = first + len(chunk_df) - 1

                        content = chunk_df.to_json(orient="records")
                        meta = {
                            "source": file_path,
                            "sheet_name": sheet_name,
                            "content_type": "table",
                            "table_index": t_idx,
                            "first_row": first,
                            "last_row": last,
                            "columns": headers,
                            "has_header": has_hdr,
                            "chunk_id": f"{sheet_name}:{t_idx}:{first}-{last}"
                        }
                        clean_meta = self._sanitize_metadata(meta)
                        documents.append(Document(page_content=content, metadata=clean_meta))
                        self.logger.debug(f"Table chunk: {sheet_name} rows {first}-{last}")
            else:
                # Unstructured text fallback
                self.logger.info(f"No tables in {sheet_name}, using UnstructuredExcelLoader")
                loader = UnstructuredExcelLoader(file_path, mode="elements", sheet_name=sheet_name)
                raw_docs = loader.load()
                text_chunks = self.text_splitter.split_documents(raw_docs)
                for idx, chunk in enumerate(text_chunks):
                    meta = {
                        "source": file_path,
                        "sheet_name": sheet_name,
                        "content_type": "text",
                        "chunk_id": f"{sheet_name}:text:{idx}"
                    }
                    clean_meta = self._sanitize_metadata(meta)
                    documents.append(Document(page_content=chunk.page_content, metadata=clean_meta))
                    self.logger.debug(f"Text chunk: {sheet_name} idx {idx}")

        self.logger.info(f"Loaded {len(documents)} chunks from {file_path}")
        return documents

    def _detect_tables(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        empty = df.index[df.isna().all(axis=1)].tolist()
        tables = []
        start = None
        empties = 0

        for i in range(len(df)):
            if i in empty:
                empties += 1
                if empties > self.max_empty_rows_between_tables and start is not None:
                    tables.append((start, i - empties))
                    start = None
                    empties = 0
            else:
                if start is None:
                    start = i
                empties = 0

        if start is not None:
            tables.append((start, len(df)))

        result = []
        for s, e in tables:
            block = df.iloc[s:e]
            if len(block) <= 1:
                continue
            header = block.iloc[0]
            is_hdr = self.is_likely_header(header)
            block.attrs["header_rows"] = 1 if is_hdr else 0
            result.append(block)
        return result

    def is_likely_header(self, row: pd.Series) -> bool:
        vals = row.dropna().astype(str)
        if vals.empty:
            return False
        str_ratio = sum(isinstance(v, str) for v in vals) / len(vals)
        uniq_ratio = len(vals.unique()) / len(vals)
        return str_ratio > 0.5 and uniq_ratio > 0.8

    def _make_unique_headers(self, headers: List[str]) -> List[str]:
        seen: Dict[str, int] = {}
        unique: List[str] = []
        for h in headers:
            key = h or "Unnamed"
            seen[key] = seen.get(key, 0) + 1
            unique.append(f"{key}_{seen[key]}" if seen[key] > 1 else key)
        return unique