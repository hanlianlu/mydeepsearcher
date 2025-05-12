import os
from abc import ABC
from typing import List

from langchain_core.documents import Document


class BaseLoader(ABC):
    def __init__(self, **kwargs):
        pass

    def load_file(self, file_path: str) -> List[Document]:
        pass
        # Return a list of Document objects which contain the text and metadata.
        # In the metadata, it's recommended to include the reference to the file.
        # e.g.
        # return [Document(page_content=..., metadata={"reference": file_path})]

    def load_directory(self, directory: str) -> List[Document]:
        documents = []
        for file in os.listdir(directory):
            for suffix in self.supported_file_types:
                if file.endswith(suffix):
                    documents.extend(self.load_file(os.path.join(directory, file)))
        return documents

    @property
    def supported_file_types(self) -> List[str]:
        pass
