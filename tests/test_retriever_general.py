from datetime import date

from denser_retriever.retriever_general import RetrieverGeneral
from denser_retriever.utils import passages_to_dict


class TestRetrieverGeneral:
    def setup_method(self):
        """setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every tests method of a class.
        """
        self.retriever = RetrieverGeneral("unit_test_index", "tests/config-test.yaml")
        self.retriever_cn = RetrieverGeneral("unit_test_cn_index", "tests/config-test-cn.yaml")

    def test_retriever_ingest(self):
        self.retriever.ingest("tests/test_data/denser_website_passages_top10.jsonl")
        self.retriever_cn.ingest("tests/test_data/cpws_passages_top10.jsonl")

    def test_retriever_retrieve(self):
        topk = 5
        query = "what is denser ai?"
        passages, docs = self.retriever.retrieve(query, {}, topk)
        assert len(passages) == 5

    def test_retriever_retrieve_cn(self):
        topk = 5
        queries = ["", "买卖合同纠纷"]
        meta_data = {"所属地区": "新昌县"}
        for query in queries:
            passages, docs = self.retriever_cn.retrieve(query, meta_data, topk)
            assert len(passages) == 3
            for passage in passages:
                assert passage["所属地区"] == "新昌县"

        queries = ["", "买卖合同纠纷"]
        meta_data = {"裁判日期": (date(2021, 10, 4), date(2021, 10, 5))}
        for query in queries:
            passages, docs = self.retriever_cn.retrieve(query, meta_data, topk)
            assert len(passages) == 2
            for passage in passages:
                assert passage["裁判日期"] == "2021-10-04" or passage["裁判日期"] == "2021-10-05"

        queries = ["", "买卖合同纠纷"]
        meta_data = {"所属地区": "新昌县", "裁判日期": (date(2021, 10, 4), date(2021, 10, 5))}
        for query in queries:
            passages, docs = self.retriever_cn.retrieve(query, meta_data, topk)
            assert len(passages) == 1
            for passage in passages:
                assert passage["所属地区"] == "新昌县"
                assert passage["裁判日期"] == "2021-10-04" or passage["裁判日期"] == "2021-10-05"
