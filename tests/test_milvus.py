import unittest
import pprint
import numpy as np
from deepsearcher.vector_db import Milvus, RetrievalResult
from deepsearcher.tools import log
from deepsearcher.tools import log
from pymilvus import Collection, utility
import pandas as pd
import json

class TestMilvus(unittest.TestCase):
    def test_milvus(self):
        d = 8
        collection = "hello_deepsearcher"
        milvus = Milvus()
        milvus.init_collection(
            dim=d,
            collection=collection,
        )
        rng = np.random.default_rng(seed=19530)
        milvus.insert_data(
        collection=collection,
        chunks=[
            RetrievalResult(
                embedding=rng.random((1, d))[0].tolist(),  # Fix applied here
                text="hello world",
                reference="local file: hi.txt",
                metadata={"a": 1},
            ),
            RetrievalResult(
                embedding=rng.random((1, d))[0].tolist(),  # Fix applied here
                text="hello milvus",
                reference="local file: hi.txt",
                metadata={"a": 1},
            ),
            RetrievalResult(
                embedding=rng.random((1, d))[0].tolist(),  # Fix applied here
                text="hello deep learning",
                reference="local file: hi.txt",
                metadata={"a": 1},
            ),
            RetrievalResult(
                embedding=rng.random((1, d))[0].tolist(),  # Fix applied here
                text="hello llm",
                reference="local file: hi.txt",
                metadata={"a": 1},
            ),
        ],
        )
        top_2 = milvus.search_data(
            collection=collection,
            vector=rng.random((1, d))[0].tolist(),  # Fix applied here
            top_k=2
        )
        log.info(pprint.pformat(top_2))

    def test_clear_collection(self):
        d = 8
        collection = "hello_deepsearcher"
        milvus = Milvus()
        milvus.init_collection(dim=d, collection=collection)  # Removed only_init_client if not supported
        milvus.clear_db(collection=collection)
        self.assertFalse(utility.has_collection(collection, timeout=5))  # Fix applied here



    def test_excel_embedding(self):
        collection = "test_excel_collection"
        milvus = Milvus()
        milvus.init_collection(dim=8, collection=collection, force_new_collection=True)
        
        # Simulate Excel data: a small DataFrame
        df = pd.DataFrame({
            "Name": ["Alice", "Bob"],
            "Age": [30, 25]
        })
        json_text = df.to_json(orient="records")
        metadata = {
            "source": "test.xlsx",
            "sheet_name": "Sheet1",
            "table_index": 0,
            "start_row": 2,
            "end_row": 3,
            "columns": ["Name", "Age"],
            "column_types": {"Name": "string", "Age": "numeric"},
            "has_missing_data": False
        }

        embedding = [0.1] * 8
        chunk = RetrievalResult(
            embedding=embedding,
            text=json_text,
            reference="test.xlsx",
            metadata=metadata
        )
        milvus.insert_data(collection=collection, chunks=[chunk])

        # Key Difference: Flush the collection
        collection_obj = Collection(name=collection)
        collection_obj.flush()

        # Query the collection
        docs = collection_obj.query(expr="", output_fields=["text", "metadata"], limit=10)

        # Expected metadata with string values
        expected_metadata = {
            "source": "test.xlsx",
            "sheet_name": "Sheet1",
            "table_index": "0",
            "start_row": "2",
            "end_row": "3",
            "columns": "['Name', 'Age']",
            "column_types": "{'Name': 'string', 'Age': 'numeric'}",
            "has_missing_data": "False"
        }

        # Assertions
        self.assertEqual(len(docs), 1, "Should retrieve exactly one document")
        retrieved_text = docs[0]["text"]
        retrieved_metadata_str = docs[0]["metadata"]
        retrieved_metadata = json.loads(retrieved_metadata_str)
        
        self.assertEqual(retrieved_text, json_text, "Text should match inserted JSON")
        self.assertIsInstance(retrieved_metadata, dict, "Metadata should be a dictionary")
        self.assertEqual(retrieved_metadata, expected_metadata, "Metadata should match inserted metadata")
    
    def test_search_excel_data(self):
        # Setup: Insert test data
        collection = "test_collection"  # Adjust to your collection name
        milvus = Milvus()
        milvus.init_collection(dim=8, collection=collection, force_new_collection=True)
        json_text1 = '[{"Name":"Alice","Age":30}]'  # "Alice" document
        json_text2 = '[{"Name":"Bob","Age":25}]'    # "Bob" document
        metadata1 = {"source": "test.xlsx", "sheet_name": "Sheet1"}
        metadata2 = {"source": "test.xlsx", "sheet_name": "Sheet1"}

        # Define embeddings
        embedding1 = [0.1] * 8  # Embedding for "Alice"
        embedding2 = [0.9] * 8  # Embedding for "Bob"

        # Create RetrievalResult objects
        chunk1 = RetrievalResult(
            embedding=embedding1,
            text=json_text1,
            reference="test.xlsx",
            metadata=metadata1
        )
        chunk2 = RetrievalResult(
            embedding=embedding2,
            text=json_text2,
            reference="test.xlsx",
            metadata=metadata2
        )

        # Insert data into Milvus
        milvus.insert_data(collection=collection, chunks=[chunk1, chunk2])

        # Search using the exact "Alice" embedding
        search_vector = embedding1  # Exact match for "Alice"
        results = milvus.search_data(collection=collection, vector=search_vector, top_k=1)

        # Assertions
        self.assertEqual(len(results), 1, "Should retrieve one result")
        self.assertEqual(results[0].text, json_text1, "Should retrieve the 'Alice' document")
        self.tearDown()

    def tearDown(self):
        milvus = Milvus()
        for collection in ["test_excel_collection", "test_search_collection"]:
            if utility.has_collection(collection):
                utility.drop_collection(collection)
if __name__ == "__main__":
        unittest.main()
