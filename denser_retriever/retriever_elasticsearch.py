from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch
from elasticsearch.helpers import BulkIndexError, bulk

from denser_retriever.retriever import Passage, Retriever
from denser_retriever.utils import get_logger

logger = get_logger(__name__)


class RetrieverElasticSearch(Retriever):
    """
    Elasticsearch Retriever
    """

    retrieve_type = "elasticsearch"
    es: Elasticsearch

    def __init__(self, index_name: str, config_path: str = "config.yaml"):
        super().__init__(index_name, config_path)
        self.drop_old = self.settings.drop_old
        self.es = Elasticsearch(
            hosts=[self.settings.keyword.es_host],
            basic_auth=(self.settings.keyword.es_user, self.settings.keyword.es_passwd),
            request_timeout=600,
        )

    def create_index(self, index_name: str):
        # Define the index settings and mappings

        logger.info(f"ES analysis {self.settings.keyword.analysis}")
        if self.settings.keyword.analysis == "default":
            settings = {
                "analysis": {"analyzer": {"default": {"type": "standard"}}},
                "similarity": {
                    "custom_bm25": {
                        "type": "BM25",
                        "k1": 1.2,
                        "b": 0.75,
                    }
                },
            }
            mappings = {
                "properties": {
                    "content": {
                        "type": "text",
                        "similarity": "custom_bm25",  # Use the custom BM25 similarity
                    },
                    "title": {
                        "type": "text",
                    },
                    "source": {
                        "type": "keyword",
                    },
                    "pid": {
                        "type": "text",
                    },
                }
            }
        else:  # ik
            settings = {
                "analysis": {
                    "analyzer": {
                        "ik_max_word": {"type": "custom", "tokenizer": "ik_max_word"},
                        "ik_smart": {"type": "custom", "tokenizer": "ik_smart"},
                    }
                },
                "similarity": {
                    "custom_bm25": {
                        "type": "BM25",
                        "k1": 1.2,
                        "b": 0.75,
                    }
                },
            }
            mappings = {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "similarity": "custom_bm25",  # Use the custom BM25 similarity
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "ik_smart",
                    },
                    "source": {
                        "type": "text",
                    },
                    "pid": {
                        "type": "text",
                    },
                }
            }

        for key in self.field_types:
            mappings["properties"][key] = self.field_types[key]

        # Create the index with the specified settings and mappings
        # if self.es.indices.exists(index=index_name):
        # self.es.indices.delete(index=index_name)
        self.es.indices.create(index=index_name, mappings=mappings, settings=settings)

    def ingest(
        self,
        passages: List[Passage],
        ids: List[str],
        batch_size: int,
        refresh_indices: bool = True,
    ) -> list[str]:
        # Check if the index exists, if not create it
        if not self.es.indices.exists(index=self.index_name):
            self.create_index(self.index_name)
        # Check if drop_old is True, if so, delete the index and recreate it
        if self.drop_old:
            self.es.indices.delete(index=self.index_name)
            self.create_index(self.index_name)

        requests = []
        batch_count = 0
        record_id = 0
        for passage, _id in zip(passages, ids):
            request = {
                "_op_type": "index",
                "_index": self.index_name,
                "content": passage.get("text", ""),
                "title": passage.get("title", ""),
                "_id": _id,
                "source": passage.get("source", ""),
                "pid": passage.get("pid", -1),
            }
            for filter in self.field_types.keys():
                v = getattr(passage, filter).strip()
                if v:
                    request[filter] = v
            ids.append(_id)
            requests.append(request)

            batch_count += 1
            record_id += 1
            if batch_count >= batch_size:
                # Index the batch
                bulk(self.es, requests)
                logger.info(f"ES ingesting {record_id}")
                batch_count = 0
                requests = []

        # Index any remaining documents
        if requests:
            bulk(self.es, requests)
            logger.info(f"ES ingesting record {record_id}")

        if refresh_indices:
            self.es.indices.refresh(index=self.index_name)

        return ids

    def retrieve(self, query_text, meta_data, query_id=None):
        assert self.es.indices.exists(index=self.index_name)

        query_dict = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "title": {
                                    "query": query_text,
                                    "boost": 2.0,  # Boost the "title" field with a higher weight
                                }
                            }
                        },
                        {"match": {"content": query_text}},
                    ],
                    "must": [],
                }
            },
            "_source": True,
        }

        for field in meta_data:
            category_or_date = meta_data.get(field)
            if category_or_date:
                if isinstance(category_or_date, tuple):
                    query_dict["query"]["bool"]["must"].append(
                        {
                            "range": {
                                field: {
                                    "gte": category_or_date[0],
                                    "lte": category_or_date[1]
                                    if len(category_or_date) > 1
                                    else category_or_date[0],  # type: ignore
                                }
                            }
                        }
                    )
                else:
                    query_dict["query"]["bool"]["must"].append(
                        {"term": {field: category_or_date}}
                    )

        res = self.es.search(
            index=self.index_name, body=query_dict, size=self.settings.keyword.topk
        )
        topk_used = min(len(res["hits"]["hits"]), self.settings.keyword.topk)
        passages = []
        for id in range(topk_used):
            _source = res["hits"]["hits"][id]["_source"]
            passage = {
                "id": res["hits"]["hits"][id]["_id"],
                "source": _source["source"],
                "text": _source["content"],
                "title": _source["title"],
                "pid": _source["pid"],
                "score": res["hits"]["hits"][id]["_score"],
            }
            for field in meta_data:
                if _source.get(field):
                    passage[field] = _source.get(field)
            passages.append(passage)
        return passages

    def get_index_mappings(self):
        mapping = self.es.indices.get_mapping(index=self.index_name)

        # The mapping response structure can be quite nested, focusing on the 'properties' section
        properties = mapping[self.index_name]["mappings"]["properties"]

        # Function to recursively extract fields and types
        def extract_fields(fields_dict, parent_name=""):
            fields = {}
            for field_name, details in fields_dict.items():
                full_field_name = (
                    f"{parent_name}.{field_name}" if parent_name else field_name
                )
                if "properties" in details:
                    fields.update(
                        extract_fields(details["properties"], full_field_name)
                    )
                else:
                    fields[full_field_name] = details.get(
                        "type", "notype"
                    )  # Default 'notype' if no type is found
            return fields

        # Extract fields and types
        all_fields = extract_fields(properties)
        return all_fields

    def get_categories(self, field, topk):
        query = {
            "size": 0,  # No actual documents are needed, just the aggregation results
            "aggs": {
                "all_categories": {
                    "terms": {
                        "field": field,
                        "size": 1000,  # Adjust this value based on the expected number of unique categories
                    }
                }
            },
        }
        response = self.es.search(index=self.index_name, body=query)
        # Extract the aggregation results
        categories = response["aggregations"]["all_categories"]["buckets"]
        if topk > 0:
            categories = categories[:topk]
        res = [category["key"] for category in categories]
        return res

    def delete(  # type: ignore[no-untyped-def]
        self,
        *,
        ids: Optional[List[str]] = None,
        query: Optional[Dict[str, Any]] = None,
        refresh_indices: bool = True,
        **delete_kwargs,
    ) -> bool:
        """Delete documents from the Elasticsearch index.

        :param ids: List of IDs of documents to delete.
        :param refresh_indices: Whether to refresh the index after deleting documents.
            Defaults to True.

        :return: True if deletion was successful.
        """
        if ids is not None and query is not None:
            raise ValueError("one of ids or query must be specified")
        elif ids is None and query is None:
            raise ValueError("either specify ids or query")

        try:
            if ids:
                body = [
                    {"_op_type": "delete", "_index": self.index_name, "_id": _id}
                    for _id in ids
                ]
                bulk(
                    self.es,
                    body,
                    refresh=refresh_indices,
                    ignore_status=404,
                    **delete_kwargs,
                )
                logger.debug(f"Deleted {len(body)} texts from index")

            else:
                self.es.delete_by_query(
                    index=self.index_name,
                    body=query,
                    refresh=refresh_indices,
                    **delete_kwargs,
                )

        except BulkIndexError as e:
            logger.error(f"Error deleting texts: {e}")
            firstError = e.errors[0].get("index", {}).get("error", {})
            logger.error(f"First error reason: {firstError.get('reason')}")
            raise e

        return True

    def delete_by_source(self, source: str, refresh_indices: bool = True):
        query = {"query": {"term": {"source": source}}}
        return self.delete(query=query, refresh_indices=refresh_indices)
