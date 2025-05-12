import array
import json
from typing import List, Optional, Union

import numpy as np

from deepsearcher.loader.splitter import Chunk
from deepsearcher.tools import log
from deepsearcher.vector_db.base import BaseVectorDB, CollectionInfo, RetrievalResult


class OracleDB(BaseVectorDB):
    """OracleDB class is a subclass of DB class."""

    client = None

    def __init__(
        self,
        user: str,
        password: str,
        dsn: str,
        config_dir: str,
        wallet_location: str,
        wallet_password: str,
        min: int = 1,
        max: int = 10,
        increment: int = 1,
        default_collection: str = "deepsearcher",
    ):
        super().__init__(default_collection)
        self.default_collection = default_collection

        import oracledb

        oracledb.defaults.fetch_lobs = False
        self.DB_TYPE_VECTOR = oracledb.DB_TYPE_VECTOR

        try:
            self.client = oracledb.create_pool(
                user=user,
                password=password,
                dsn=dsn,
                config_dir=config_dir,
                wallet_location=wallet_location,
                wallet_password=wallet_password,
                min=min,
                max=max,
                increment=increment,
            )
            log.color_print(f"Connected to Oracle database at {dsn}")
        except Exception as e:
            log.critical(f"Failed to connect to Oracle database at {dsn}")
            log.critical(f"Oracle database error in init: {e}")
            raise

    def numpy_converter_in(self, value):
        """Convert numpy array to array.array"""
        if value.dtype == np.float64:
            dtype = "d"
        elif value.dtype == np.float32:
            dtype = "f"
        else:
            dtype = "b"
        return array.array(dtype, value)

    def input_type_handler(self, cursor, value, arraysize):
        """Set the type handler for the input data"""
        if isinstance(value, np.ndarray):
            return cursor.var(
                self.DB_TYPE_VECTOR,
                arraysize=arraysize,
                inconverter=self.numpy_converter_in,
            )

    def numpy_converter_out(self, value):
        """Convert array.array to numpy array"""
        if value.typecode == "b":
            dtype = np.int8
        elif value.typecode == "f":
            dtype = np.float32
        else:
            dtype = np.float64
        return np.array(value, copy=False, dtype=dtype)

    def output_type_handler(self, cursor, metadata):
        """Set the type handler for the output data"""
        if metadata.type_code is self.DB_TYPE_VECTOR:
            return cursor.var(
                metadata.type_code,
                arraysize=cursor.arraysize,
                outconverter=self.numpy_converter_out,
            )

    def query(self, sql: str, params: dict = None) -> Union[dict, None]:
        with self.client.acquire() as connection:
            connection.inputtypehandler = self.input_type_handler
            connection.outputtypehandler = self.output_type_handler
            with connection.cursor() as cursor:
                try:
                    if log.dev_mode:
                        print("sql:\n", sql)
                    # log.debug("def query:"+params)
                    # print("sql:\n",sql)
                    # print("params:\n",params)
                    cursor.execute(sql, params)
                except Exception as e:
                    log.critical(f"Oracle database error in query: {e}")
                    raise
                columns = [column[0].lower() for column in cursor.description]
                rows = cursor.fetchall()
                if rows:
                    data = [dict(zip(columns, row)) for row in rows]
                else:
                    data = []
                if log.dev_mode:
                    print("data:\n", data)
                return data
            # self.client.drop(connection)

    def execute(self, sql: str, data: Union[list, dict] = None):
        try:
            with self.client.acquire() as connection:
                connection.inputtypehandler = self.input_type_handler
                connection.outputtypehandler = self.output_type_handler
                with connection.cursor() as cursor:
                    # print("sql:\n",sql)
                    # print("data:\n",data)
                    if data is None:
                        cursor.execute(sql)
                    else:
                        cursor.execute(sql, data)
                    connection.commit()
        except Exception as e:
            log.critical(f"Oracle database error in execute: {e}")
            log.error("ERROR sql:\n" + sql)
            log.error("ERROR data:\n" + data)
            raise

    def has_collection(self, collection: str = "deepsearcher"):
        SQL = SQL_TEMPLATES["has_collection"]
        params = {"collection": collection}
        res = self.query(SQL, params)
        if res:
            if res[0]["rowcnt"] > 0:
                return True
            else:
                return False
        else:
            return False

    def has_table(self, collection: str = "deepsearcher"):
        SQL = SQL_TEMPLATES["has_table"]
        res = self.query(SQL)
        if res:
            if res[0]["rowcnt"] > 0:
                return True
            else:
                return False
        else:
            return False

    def create_tables(self):
        SQL = SQL_TEMPLATES["ddl"]
        try:
            self.execute(SQL)
            log.color_print("Created table DEEPSEARCHER in Oracle database")
        except Exception:
            log.critical("Failed to create table DEEPSEARCHER in Oracle database")
            raise

    def drop_collection(self, collection: str = "deepsearcher"):
        try:
            SQL = SQL_TEMPLATES["drop_collection"]
            params = {"collection": collection}
            self.execute(SQL, params)
            log.color_print(f"Collection {collection} dropped")
        except Exception as e:
            log.critical(f"fail to drop collection, error info: {e}")
            raise

    def insertone(self, data):
        SQL = SQL_TEMPLATES["insert"]
        self.execute(SQL, data)
        log.debug("insert done!")

    def searchone(
        self, collection: Optional[str], vector: Union[np.array, List[float]], top_k: int = 5
    ):
        log.debug("def searchone:" + collection)
        try:
            if isinstance(vector, List):
                vector = np.array(vector)
            embedding_string = "[" + ", ".join(map(str, vector.tolist())) + "]"
            dimension = vector.shape[0]
            dtype = str(vector.dtype).upper()

            SQL = SQL_TEMPLATES["search"].format(dimension=dimension, dtype=dtype)
            max_distance = 0.8
            params = {
                "collection": collection,
                "embedding_string": embedding_string,
                "top_k": top_k,
                "max_distance": max_distance,
            }
            res = self.query(SQL, params)
            if res:
                return res
            else:
                return []
        except Exception as e:
            log.critical(f"fail to search data, error info: {e}")
            raise

    def init_collection(
        self,
        dim: int,
        collection: Optional[str] = "deepsearcher",
        description: Optional[str] = "",
        force_new_collection: bool = False,
        text_max_length: int = 65_535,
        reference_max_length: int = 2048,
        metric_type: str = "L2",
        *args,
        **kwargs,
    ):
        if not collection:
            collection = self.default_collection
        if description is None:
            description = ""
        try:
            has_table = self.has_table()
            if not has_table:
                self.create_table()
        except Exception as e:
            log.critical(f"fail to init db for oracle, error info: {e}")

        try:
            has_collection = self.has_collection(collection)
            if force_new_collection and has_collection:
                self.drop_collection(collection)
            elif has_collection:
                return
        except Exception as e:
            log.critical(f"fail to init db for milvus, error info: {e}")
            raise

    def insert_data(
        self,
        collection: Optional[str],
        chunks: List[Chunk],
        batch_size: int = 256,
        *args,
        **kwargs,
    ):
        if not collection:
            collection = self.default_collection

        datas = []
        for chunk in chunks:
            _data = {
                "embedding": self.numpy_converter_in(np.array(chunk.embedding)),
                "text": chunk.text,
                "reference": chunk.reference,
                "metadata": json.dumps(chunk.metadata),
                "collection": collection,
            }
            datas.append(_data)

        batch_datas = [datas[i : i + batch_size] for i in range(0, len(datas), batch_size)]
        try:
            for batch_data in batch_datas:
                for _data in batch_data:
                    self.insertone(data=_data)
            log.color_print(f"Successfully insert {len(datas)} data")
        except Exception as e:
            log.critical(f"fail to insert data, error info: {e}")
            raise

    def search_data(
        self,
        collection: Optional[str],
        vector: Union[np.array, List[float]],
        top_k: int = 5,
        *args,
        **kwargs,
    ) -> List[RetrievalResult]:
        if not collection:
            collection = self.default_collection
        try:
            # print("def search_data:",collection)
            # print("def search_data:",type(vector))
            search_results = self.searchone(collection=collection, vector=vector, top_k=top_k)
            # print("def search_data: search_results",search_results)

            return [
                RetrievalResult(
                    embedding=b["embedding"],
                    text=b["text"],
                    reference=b["reference"],
                    score=b["distance"],
                    metadata=json.loads(b["metadata"]),
                )
                for b in search_results
            ]
        except Exception as e:
            log.critical(f"fail to search data, error info: {e}")
            raise
            # return []

    def list_collections(self, *args, **kwargs) -> List[CollectionInfo]:
        collection_infos = []
        try:
            SQL = SQL_TEMPLATES["list_collections"]
            log.debug("def list_collections:" + SQL)
            collections = self.query(SQL)
            if collections:
                for collection in collections:
                    description = ""
                    collection_infos.append(
                        CollectionInfo(
                            collection_name=collection["collection"],
                            description=description,
                        )
                    )
                return collection_infos
            else:
                log.critical("No collections found")
        except Exception as e:
            log.critical(f"fail to list collections, error info: {e}")
            raise

    def clear_db(self, collection: str = "deepsearcher", *args, **kwargs):
        if not collection:
            collection = self.default_collection
        try:
            self.client.drop_collection(collection)
        except Exception as e:
            log.warning(f"fail to clear db, error info: {e}")
            raise


SQL_TEMPLATES = {
    "ddl": """CREATE TABLE DEEPSEARCHER (    
        id INT generated by default as identity primary key,
        collection varchar(256),
        embedding VECTOR,
        text CLOB,
        reference varchar(4000),
        metadata CLOB,
        status NUMBER DEFAULT 1,
        createtime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updatetime TIMESTAMP DEFAULT NULL)""",
    "has_table": "SELECT count(*) as rowcnt FROM all_tables WHERE table_name='DEEPSEARCHER'",
    "has_collection": "select count(*) as rowcnt from DEEPSEARCHER where collection=:collection and status=1",
    "list_collections": "select distinct collection from DEEPSEARCHER where status=1",
    "drop_collection": "update DEEPSEARCHER set status=0 where collection=:collection and status=1",
    "insert": """INSERT INTO DEEPSEARCHER (collection,embedding,text,reference,metadata) 
        values (:collection,:embedding,:text,:reference,:metadata)""",
    "search": """SELECT * FROM 
        (SELECT t.*,
            VECTOR_DISTANCE(t.embedding,vector(:embedding_string,{dimension},{dtype}),COSINE) as distance
        FROM DEEPSEARCHER t WHERE t.collection=:collection)
        WHERE distance<:max_distance ORDER BY distance ASC FETCH FIRST :top_k ROWS ONLY""",
}
