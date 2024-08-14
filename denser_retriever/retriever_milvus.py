import json
import os
import pickle
from datetime import datetime
from typing import List, Optional, Union
from uuid import uuid4

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    connections,
    utility,
)
from sentence_transformers import SentenceTransformer

from denser_retriever.retriever import Passage, Retriever
from denser_retriever.settings import Vector
from denser_retriever.utils import get_logger

logger = get_logger(__name__)


class RetrieverMilvus(Retriever):
    """
    Milvus Retriever
    """

    retrieve_type = "milvus"
    config: Vector

    def __init__(self, index_name: str, config_path: str = "config.yaml"):
        super().__init__(index_name, config_path)
        self.index_name = index_name
        self.drop_old = self.settings.drop_old
        self.config = self.settings.vector

        self.col = None
        self.source_max_length = 500
        self.title_max_length = 500
        self.text_max_length = 8000
        self.field_max_length = 500
        self.model = SentenceTransformer(self.config.emb_model, trust_remote_code=True)

        connection_args = {
            "host": self.config.milvus_host,
            "port": self.config.milvus_port,
            "user": self.config.milvus_user,
            "password": self.config.milvus_passwd,
        }
        self.alias = self._create_connection_alias(connection_args)
        self.col: Optional[Collection] = None

        # Grab the existing collection if it exists
        if utility.has_collection(self.index_name, using=self.alias):
            self.col = Collection(
                self.index_name,
                using=self.alias,
            )
        # If need to drop old, drop it
        if self.drop_old and isinstance(self.col, Collection):
            self.col.drop()
            self.col = None

        self._create_collection()

    def _create_connection_alias(self, connection_args: dict) -> str:
        """Create the connection to the Milvus server."""
        from pymilvus import MilvusException, connections

        # Grab the connection arguments that are used for checking existing connection
        host: str = connection_args.get("host", None)
        port: Union[str, int] = connection_args.get("port", None)
        address: str = connection_args.get("address", None)
        uri: str = connection_args.get("uri", None)
        user = connection_args.get("user", None)

        # Order of use is host/port, uri, address
        if host is not None and port is not None:
            given_address = str(host) + ":" + str(port)
        elif uri is not None:
            if uri.startswith("https://"):
                given_address = uri.split("https://")[1]
            elif uri.startswith("http://"):
                given_address = uri.split("http://")[1]
            else:
                given_address = uri  # Milvus lite
        elif address is not None:
            given_address = address
        else:
            given_address = None
            logger.debug("Missing standard address type for reuse attempt")

        # User defaults to empty string when getting connection info
        if user is not None:
            tmp_user = user
        else:
            tmp_user = ""

        # If a valid address was given, then check if a connection exists
        if given_address is not None:
            for con in connections.list_connections():
                addr = connections.get_connection_addr(con[0])
                if (
                    con[1]
                    and ("address" in addr)
                    and (addr["address"] == given_address)
                    and ("user" in addr)
                    and (addr["user"] == tmp_user)
                ):
                    logger.debug("Using previous connection: %s", con[0])
                    return con[0]

        # Generate a new connection if one doesn't exist
        alias = uuid4().hex
        try:
            connections.connect(alias=alias, **connection_args)
            logger.debug("Created new connection using: %s", alias)
            return alias
        except MilvusException as e:
            logger.error("Failed to create new connection using: %s", alias)
            raise e

    def _create_collection(self):
        # Create general schema
        fields = [
            FieldSchema(
                name="uid",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=100,
            ),
            FieldSchema(
                name="source", dtype=DataType.VARCHAR, max_length=self.source_max_length
            ),
            FieldSchema(
                name="title", dtype=DataType.VARCHAR, max_length=self.title_max_length
            ),
            FieldSchema(
                name="text", dtype=DataType.VARCHAR, max_length=self.text_max_length
            ),
            FieldSchema(name="pid", dtype=DataType.INT64),
            FieldSchema(
                name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=self.config.emb_dims
            ),
        ]
        # Create fields for each metadata field
        for key in self.field_types:
            internal_key = self.field_internal_names[key]
            # both category and date type (unix timestamp) use INT64 type
            fields.append(
                FieldSchema(
                    name=internal_key,
                    dtype=DataType.INT64,
                    max_length=self.field_max_length,
                ),
            )
            self.field_cat_to_id[key] = {}
            self.field_id_to_cat[key] = []

        # Create schema for the collection
        schema = CollectionSchema(fields, "Milvus schema")
        try:
            self.col = Collection(
                self.index_name,
                schema,
                consistency_level="Strong",
                using=self.alias,
            )
        except MilvusException as e:
            logger.error(
                "Failed to create collection: %s error: %s", self.index_name, e
            )
            raise e

    def connect_index(self):
        connections.connect(
            "default",
            host=self.config.milvus_host,
            port=self.config.milvus_port,
            user=self.config.milvus_user,
            password=self.config.milvus_passwd,
        )
        has = utility.has_collection(self.index_name)
        assert has is True
        fields = [
            FieldSchema(
                name="uid",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=100,
            ),
            FieldSchema(
                name="source", dtype=DataType.VARCHAR, max_length=self.source_max_length
            ),
            FieldSchema(
                name="title", dtype=DataType.VARCHAR, max_length=self.title_max_length
            ),
            FieldSchema(
                name="text", dtype=DataType.VARCHAR, max_length=self.text_max_length
            ),
            FieldSchema(name="pid", dtype=DataType.VARCHAR),
            FieldSchema(
                name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=self.config.emb_dims
            ),
        ]

        for key in self.field_types:
            internal_key = self.field_internal_names[key]
            # both category and date type (unix timestamp) use INT64 type
            fields.append(
                FieldSchema(
                    name=internal_key,
                    dtype=DataType.INT64,
                    max_length=self.field_max_length,
                ),
            )

        schema = CollectionSchema(fields, "Milvus schema")
        self.col = Collection(self.index_name, schema, consistency_level="Strong")
        logger.info("Loading milvus index")
        self.col.load()

        exp_dir = os.path.join(self.settings.output_prefix, f"exp_{self.index_name}")
        fields_file = os.path.join(exp_dir, "milvus_fields.json")
        if os.path.isfile(fields_file):
            with open(fields_file, "r") as file:
                self.field_cat_to_id, self.field_id_to_cat = json.load(file)

    def generate_embedding(self, texts, query=False):
        if query and not self.config.one_model:
            embeddings = self.model.encode(texts, prompt_name="query")
            # embeddings = self.model.encode(texts, prompt="Represent this sentence for searching relevant passages:")
        else:
            embeddings = self.model.encode(texts)
        return embeddings

    def ingest(
        self,
        passages: List[Passage],
        ids: List[str],
        batch_size: int,
    ) -> list[str]:
        batch = []
        uid_list, sources, titles, texts, pid_list = [], [], [], [], []
        fields_list = [[] for _ in self.field_types.keys()]
        failed_batches = []  # To store information about failed batches
        records_per_file = []
        for passage, id in zip(passages, ids):
            batch.append(
                passage["title"][: self.title_max_length - 10]
                + " "
                + passage["text"][:2000]
            )
            uid_list.append(id)
            sources.append(passage.get("source", "")[: self.source_max_length - 10])
            titles.append(passage.get("title", "")[: self.title_max_length - 10])
            texts.append(
                passage.get("text", "")[: self.text_max_length - 1000]
            )  # buffer
            pid_list.append(passage.get("pid", -1))

            for i, field in enumerate(self.field_types.keys()):
                category_or_date_str = getattr(passage, field).strip()
                if category_or_date_str:
                    type = self.field_types[field]["type"]
                    if type == "date":
                        date_obj = datetime.strptime(category_or_date_str, "%Y-%m-%d")
                        unix_time = int(date_obj.timestamp())
                        fields_list[i].append(unix_time)
                    else:
                        if category_or_date_str not in self.field_cat_to_id[field]:
                            self.field_cat_to_id[field][category_or_date_str] = len(
                                self.field_cat_to_id[field]
                            )
                            self.field_id_to_cat[field].append(category_or_date_str)
                        fields_list[i].append(
                            self.field_cat_to_id[field][category_or_date_str]
                        )
                else:
                    fields_list[i].append(-1)

            if len(batch) == batch_size:
                embeddings = self.generate_embedding(batch)
                record = [
                    uid_list,
                    sources,
                    titles,
                    texts,
                    pid_list,
                    np.array(embeddings),
                ]
                record += fields_list
                try:
                    self.col.insert(record)
                except Exception as e:
                    logger.error(f"Milvus index insert error at record {id} - {e}")
                    failed_batches.append(
                        {
                            "sources": sources,
                            "pids": pid_list,
                            "batch": batch,
                        }
                    )

                records_per_file.append(record)
                if len(records_per_file) == 1000:
                    with open(f"{self.index_name}_{id}.pkl", "wb") as file:
                        pickle.dump(records_per_file, file)
                    records_per_file = []
                self.col.flush()
                logger.info(f"Milvus vector DB ingesting {id}")

                batch = []
                uid_list, sources, titles, texts, pid_list = [], [], [], [], []
                fields_list = [[] for _ in self.field_types.keys()]

        if len(batch) > 0:
            embeddings = self.generate_embedding(batch)
            record = [
                uid_list,
                sources,
                titles,
                texts,
                pid_list,
                np.array(embeddings),
            ]
            record += fields_list
            try:
                self.col.insert(record)
            except Exception as e:
                logger.error(f"Milvus index insert error at record {id} - {e}")
                failed_batches.append(
                    {
                        "sources": sources,
                        "pids": pid_list,
                        "batch": batch,
                    }
                )
            with open(f"{self.index_name}_{id}.pkl", "wb") as file:
                pickle.dump(records_per_file, file)
            self.col.flush()
            logger.info(f"Milvus vector DB ingesting {id}")

        # Save failed batches to a JSONL file
        failure_output_file = f"{self.index_name}.failed"
        with open(failure_output_file, "w") as fout:
            for failed_batch in failed_batches:
                for source, pid, record in zip(
                    failed_batch["sources"], failed_batch["pids"], failed_batch["batch"]
                ):
                    json.dump({"source": source, "pid": pid, "data": record}, fout)
                    fout.write("\n")

        index = {
            "index_type": "FLAT",
            "metric_type": "L2",
        }

        self.col.create_index("embeddings", index)
        self.col.load()
        exp_dir = os.path.join(self.settings.output_prefix, f"exp_{self.index_name}")
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        fields_file = os.path.join(exp_dir, "milvus_fields.json")
        with open(fields_file, "w") as file:
            json.dump(
                [self.field_cat_to_id, self.field_id_to_cat],
                file,
                ensure_ascii=False,
                indent=4,
            )

        return ids

    def retrieve(self, query_text, meta_data, query_id=None):
        if not self.col:
            self.connect_index()
        embeddings = self.generate_embedding([query_text], query=True)
        query_embedding = np.array(embeddings)

        exprs = []
        for field in meta_data:
            category_or_date_str = meta_data.get(field)
            internal_field = self.field_internal_names.get(field)
            type = self.field_types[field]["type"]
            if type == "date":
                if len(category_or_date_str) == 2:
                    start_unix_time = int(
                        datetime.combine(
                            category_or_date_str[0], datetime.min.time()
                        ).timestamp()
                    )
                    end_unix_time = int(
                        datetime.combine(
                            category_or_date_str[1], datetime.min.time()
                        ).timestamp()
                    )
                    exprs.append(f"{internal_field} >= {start_unix_time}")
                    exprs.append(f"{internal_field} <= {end_unix_time}")
                else:
                    unix_time = int(
                        datetime.combine(
                            category_or_date_str[0], datetime.min.time()
                        ).timestamp()
                    )
                    exprs.append(f"{internal_field} == {unix_time}")
            else:
                category_id = self.field_cat_to_id[field].get(category_or_date_str)
                if category_id is not None:
                    exprs.append(f"{internal_field}=={category_id}")
        expr_str = " and ".join(exprs)
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        result = self.col.search(
            query_embedding,
            "embeddings",
            search_params,
            limit=self.config.topk,
            expr=expr_str,
            output_fields=["source", "title", "text", "pid", "uid"]
            + list(self.field_internal_names.values()),
        )

        topk_used = min(len(result[0]), self.config.topk)
        passages = []
        for id in range(topk_used):
            assert len(result) == 1
            hit = result[0][id]
            passage = {
                "id": hit.entity.uid,
                "source": hit.entity.source,
                "text": hit.entity.text,
                "title": hit.entity.title,
                "pid": hit.entity.pid,
                "score": -hit.entity.distance,
            }
            for field in self.field_types.keys():
                internal_field = self.field_internal_names[field]
                cat_id_or_unix_time = hit.entity.__dict__["fields"].get(internal_field)
                type = self.field_types[field]["type"]
                if type == "date":
                    date = datetime.utcfromtimestamp(cat_id_or_unix_time).strftime(
                        "%Y-%m-%d"
                    )
                    passage[field] = date
                else:
                    passage[field] = self.field_id_to_cat[field][cat_id_or_unix_time]
            passages.append(passage)

        return passages

    def delete(  # type: ignore[no-untyped-def]
        self,
        ids: Optional[List[str]] = None,
        expr: Optional[str] = None,
        **kwargs: str,
    ):
        """Delete by vector ID or boolean expression.
        Refer to [Milvus documentation](https://milvus.io/docs/delete_data.md)
        for notes and examples of expressions.

        Args:
            ids: List of ids to delete.
            expr: Boolean expression that specifies the entities to delete.
            kwargs: Other parameters in Milvus delete api.
        """
        if isinstance(ids, list) and len(ids) > 0:
            if expr is not None:
                logger.warning(
                    "Both ids and expr are provided. " "Ignore expr and delete by ids."
                )
            expr = f"uid in {ids}"
        else:
            assert isinstance(
                expr, str
            ), "Either ids list or expr string must be provided."
        return self.col.delete(expr=expr, **kwargs)

    def delete_by_source(self, source: str, **kwargs: str):
        return self.delete(expr=f'source=="{source}"', **kwargs)
