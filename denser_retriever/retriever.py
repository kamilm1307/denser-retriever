import os
from abc import ABC, abstractmethod
from typing import Any, List


from denser_retriever.settings import RetrieverSettings


Passage = Any


class Retriever(ABC):
    """
    Base class for all Retriever
    """

    settings: RetrieverSettings

    def __init__(self, index_name: str, config_path: str = "config.yaml"):
        self.settings = RetrieverSettings.from_yaml(config_path)
        self.index_name = index_name
        self.retrieve_type = None
        self.field_types = {}
        self.field_internal_names = {}
        self.field_cat_to_id = {}
        self.field_id_to_cat = {}
        fields = self.settings.fields
        if fields:
            for f in fields:
                comps = f.split(":")
                assert len(comps) == 2 or len(comps) == 3
                self.field_types[comps[0]] = {"type": comps[-1]}
                if len(comps) == 3:
                    self.field_internal_names[comps[0]] = comps[1]
                self.field_cat_to_id[comps[0]] = {}
                self.field_id_to_cat[comps[0]] = []
        output_prefix = self.settings.output_prefix
        self.exp_dir = os.path.join(output_prefix, f"exp_{index_name}")
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

    @abstractmethod
    def ingest(self, passages: List[Passage], **kwargs: Any):
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int, **kwargs: Any):
        return None

    @abstractmethod
    def delete(self, ids: List[str], **kwargs: Any):
        pass

    @abstractmethod
    def delete_by_source(self, source: str, **kwargs: Any):
        pass
