from typing import Any, Dict, List, Optional, Sequence

from llama_index import StorageContext, ServiceContext
from llama_index.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.data_structs.data_structs import IndexDict
from llama_index.indices.base import BaseIndex
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.schema import BaseNode, NodeWithScore
from llama_index.storage.docstore.types import RefDocInfo

from threading import Lock

import json
import sqlite3

EMBED_CACHE_PATH = ".embedding.db"
DISABLE_CACHE = False


class MultiVectorIndexRetriever(BaseRetriever):
    _cache_lock = Lock()

    def search_from_cache(self, model_name, embedding_strs):
        if DISABLE_CACHE:
            return None
        with self._cache_lock:
            conn = sqlite3.connect(EMBED_CACHE_PATH)
            cur = conn.cursor()
            cur.execute(
                "CREATE TABLE IF NOT EXISTS records(model_name TEXT, embedding_strs TEXT, embedding TEXT, UNIQUE(model_name, embedding_strs))"
            )
            cur.execute(
                "SELECT * FROM records WHERE model_name = ? AND embedding_strs = ?",
                (model_name, str(embedding_strs)),
            )
            data = cur.fetchone()
            conn.close()
            if data:
                (_, _, embedding) = data
                return json.loads(embedding)
        return None

    def insert_to_cache(self, model_name, embedding_strs, embedding):
        with self._cache_lock:
            conn = sqlite3.connect(EMBED_CACHE_PATH)
            cur = conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO records (model_name, embedding_strs, embedding) VALUES (?,?,?)",
                (model_name, str(embedding_strs), json.dumps(embedding)),
            )
            conn.commit()
            conn.close()

    def __init__(
        self,
        indexes: [VectorStoreIndex],
        embed_model_name: str = "",
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        **kwargs: Any,
    ) -> None:
        sub_retrievers = []
        for index in indexes:
            retrieve = VectorIndexRetriever(
                index, similarity_top_k=similarity_top_k, **kwargs
            )
            sub_retrievers.append(retrieve)
        self._embed_model_name = embed_model_name
        self._embed_model = indexes[0].service_context.embed_model
        self._is_embedding_query = indexes[0].vector_store.is_embedding_query
        self._retrievers = sub_retrievers
        self._similarity_top_k = similarity_top_k

    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        sub_retrievers = []
        if self._is_embedding_query:
            if query_bundle.embedding is None:
                embedding_from_cache = self.search_from_cache(
                    self._embed_model_name, query_bundle.embedding_strs
                )
                query_bundle.embedding = (
                    embedding_from_cache
                    if embedding_from_cache
                    else (
                        self._embed_model.get_agg_embedding_from_queries(
                            query_bundle.embedding_strs
                        )
                    )
                )
                self.insert_to_cache(
                    self._embed_model_name,
                    query_bundle.embedding_strs,
                    query_bundle.embedding,
                )
        nodes_with_score = []
        for retriever in self._retrievers:
            nodes_with_score += retriever._get_nodes_with_embeddings(query_bundle)

        nodes_with_score.sort(key=lambda x: x.score, reverse=True)
        output = nodes_with_score[: self._similarity_top_k]
        return output


class MultiVectorStore(BaseIndex[IndexDict]):
    def __init__(
        self,
        indexes: Optional[VectorStoreIndex] = None,
        service_context: Optional[ServiceContext] = None,
        storage_context: Optional[StorageContext] = None,
        use_async: bool = False,
        store_nodes_override: bool = False,
        show_progress: bool = False,
        embed_model_name: str = "",
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._indexes = indexes
        self._use_async = use_async
        self._store_nodes_override = store_nodes_override
        self._embed_model_name = embed_model_name
        nodes = None
        index_struct = indexes[0].index_struct
        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            storage_context=storage_context,
            show_progress=show_progress,
            **kwargs,
        )

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        return MultiVectorIndexRetriever(
            self._indexes, self._embed_model_name, **kwargs
        )

    def _build_index_from_nodes(self, nodes: Sequence[BaseNode]) -> IndexDict:
        pass

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        pass

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        pass

    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        pass
