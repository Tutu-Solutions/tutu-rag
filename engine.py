import streamlit as st

import os
import json
import uuid
import yaml

from pathlib import Path

## LlamaIndex Import
from llama_index.legacy import ServiceContext, StorageContext
from llama_index.legacy import SimpleDirectoryReader
from llama_index.legacy.indices.vector_store import VectorStoreIndex

## LlamaIndex Debug
from llama_index.legacy.callbacks import CallbackManager, LlamaDebugHandler, CBEventType

## Prompts Import
from llama_index.legacy.prompts.prompts import QuestionAnswerPrompt
from llama_index.legacy.prompts.prompts import RefinePrompt

## Local Import
from index_module import get_multi_vector_index
from index_module import MULTI_STORAGE_DIR
from index_module import add_record_to_storage_config

## LLM Cache Import
import langchain
from langchain.cache import SQLiteCache

## Chat LLM Import
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import BedrockChat
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAI
import bigframes.dataframe
from langchain_google_vertexai import ChatVertexAI

## AWS Bedrock Import
import boto3

## Anthropic LLM Import
from langchain_anthropic import ChatAnthropic

## IBM LLM Import
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import (
    GenTextParamsMetaNames as GenParams,
)
from ibm_watson_machine_learning.foundation_models.extensions.langchain import (
    WatsonxLLM,
)

import settings
from keys import *

from lc_engine import LCBridge


@st.cache_resource
def get_bedrock_runtime(aws_access_key_id, aws_secret_access_key, region_name):
    return boto3.client(
        "bedrock-runtime",
        region_name=region_name.split(",", 1)[1],
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


class AuthRecord(object):
    def __init__(self, api_key, g_api_key, i_api_key, i_project_id, i_auth_endpoint):
        self.api_key = api_key
        self.g_api_key = g_api_key
        self.i_api_key = i_api_key
        self.i_project_id = i_project_id
        self.i_auth_endpoint = i_auth_endpoint

    def add_anthropic_key(self, key):
        self.anthropic_key = key

    def add_aws_key(self, aws_access_key_id, aws_secret_access_key, region_name):
        self.bedrock_runtime = get_bedrock_runtime(
            aws_access_key_id, aws_secret_access_key, region_name
        )


def get_llm_model_config():
    return yaml.safe_load(Path(settings.LLM_CONFIG_FILE).read_text(encoding="utf-8"))


def get_llm_model_categories():
    config = get_llm_model_config()
    categories = set()
    for k in config:
        if "hidden" in config[k] and config[k]["hidden"]:
            continue
        category = config[k]["category"]
        categories.add(category)
    return categories


def get_llm_model_by_key(key):
    config = get_llm_model_config()
    return config[key]


def get_llm_model_by_categories(category):
    config = get_llm_model_config()
    ret = []
    for k in config:
        if "hidden" in config[k] and config[k]["hidden"]:
            continue
        if config[k]["category"] == category:
            ret.append(k)
    return ret


def get_embed_model_config():
    return yaml.safe_load(Path(settings.EMBED_CONFIG_FILE).read_text(encoding="utf-8"))


def get_embed_model_keys():
    config = get_embed_model_config()
    ret = []
    for k in config:
        if "hidden" in config[k] and config[k]["hidden"]:
            continue
        ret.append(k)
    return ret


def get_embed_model_by_key(key):
    config = get_embed_model_config()
    return config[key]


@st.cache_resource(max_entries=1)
def get_local_embed_model(model_name, retrieval: bool):
    from langchain.embeddings import HuggingFaceEmbeddings
    from typing import Any, List

    class HuggingFaceQueryEmbeddings(HuggingFaceEmbeddings):
        retrieve: bool = False

        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            prepend = "passage: " if self.retrieve else "query :"
            return super().embed_documents([prepend + text for text in texts])

        def embed_query(self, text: str) -> List[float]:
            prepend = "passage: " if self.retrieve else "query :"
            return super().embed_query(prepend + text)

    if model_name.startswith("intfloat/multilingual-e5"):
        return HuggingFaceQueryEmbeddings(model_name=model_name, retrieve=retrieval)

    return HuggingFaceEmbeddings(model_name=model_name)


@st.cache_resource(hash_funcs={AuthRecord: lambda x: str(x.__dict__)})
def get_api_embed_model(service, model_name, auth_obj, retrieval):
    if service == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=auth_obj.g_api_key,
            task_type="retrieval_query" if retrieval else "retrieval_document",
        )
    elif service == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model_name, openai_api_key=auth_obj.api_key)
    elif service == "Cohere":
        from langchain_cohere import CohereEmbeddings

        return CohereEmbeddings(
            model=model_name,
            input_type="search_query" if retrieval else "search_document",
        )


def get_embed_model(embed_model, auth_obj: AuthRecord, retrieval: bool = False):
    from langchain.storage import LocalFileStore
    from langchain.embeddings import CacheBackedEmbeddings
    from llama_index.legacy.embeddings.langchain import LangchainEmbedding

    embed_store = LocalFileStore("./embed_cache/")

    service = embed_model["service"]
    model_name = embed_model["model"]

    if service == "local":
        underlying_embed_model = get_local_embed_model(model_name, retrieval)
    else:
        underlying_embed_model = get_api_embed_model(
            service, model_name, auth_obj, retrieval
        )

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embed_model, embed_store, namespace=model_name.replace("@", "_")
    )

    return (LangchainEmbedding(cached_embedder), model_name)


@st.cache_resource(hash_funcs={AuthRecord: lambda x: str(x.__dict__)})
def get_llm_model(llm_model_entry, auth_obj):
    llm_service = llm_model_entry["service"]
    llm_model = llm_model_entry["model"]
    if llm_service == "Google":
        return GoogleGenerativeAI(model=llm_model, google_api_key=auth_obj.g_api_key)
    elif llm_service == "Vertex":
        return ChatVertexAI(model_name=llm_model)
    elif llm_service == "Anthropic":
        return ChatAnthropic(model=llm_model, anthropic_api_key=auth_obj.anthropic_key)
    elif llm_service == "Amazon":
        return BedrockChat(client=auth_obj.bedrock_runtime, model_id=llm_model)
    elif llm_service == "Cohere":
        from langchain_cohere import ChatCohere

        return ChatCohere(modl=llm_model)
    elif llm_service == "IBM":
        credentials = {
            "url": auth_obj.i_auth_endpoint.split(",", 1)[1],
            "apikey": auth_obj.i_api_key,
        }
        project_id = auth_obj.i_project_id

        params = {
            GenParams.MAX_NEW_TOKENS: 512,
            GenParams.MIN_NEW_TOKENS: 0,
            GenParams.DECODING_METHOD: "greedy",
            GenParams.REPETITION_PENALTY: 1,
            GenParams.TRUNCATE_INPUT_TOKENS: 3000,
        }
        model = Model(
            model_id=llm_model,
            credentials=credentials,
            project_id=project_id,
            params=params,
        )

        return WatsonxLLM(model=model)

    else:
        return ChatOpenAI(model_name=llm_model, openai_api_key=auth_obj.api_key)


def get_prompt(prompt_id):
    prompt_dict = yaml.safe_load(Path("prompts.yml").read_text(encoding="utf-8"))
    prompt = "\n".join(prompt_dict[prompt_id])
    if settings.DEBUG_MODE:
        print(prompt)
    return prompt


@st.cache_resource(hash_funcs={AuthRecord: lambda x: str(x.__dict__)})
def get_index(
    auth_obj,
    embed_model_entry,
    storage_dir,
    selected_names,
    names_tagged,
):
    return get_index_with_llm(
        auth_obj, embed_model_entry, storage_dir, selected_names, names_tagged
    )


def get_index_with_llm(
    auth_obj,
    embed_model_entry,
    storage_dir,
    selected_names,
    names_tagged,
    llm=None,
):
    (embed_model, embed_model_name) = get_embed_model(
        embed_model_entry, auth_obj, retrieval=True
    )

    # TODO: Pass in Callback Manager from top level
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])

    service_context = ServiceContext.from_defaults(
        embed_model=embed_model, callback_manager=callback_manager, llm=llm
    )

    return (
        get_multi_vector_index(
            selected_names, storage_dir, service_context, embed_model_name, names_tagged
        ),
        callback_manager,
        llama_debug,
    )


def get_query_engine(
    auth_obj,
    llm_model_key,
    embed_model_key,
    storage_dir,
    selected_names,
    names_tagged=False,
):
    embed_model_entry = get_embed_model_by_key(embed_model_key)
    (embed_model, embed_model_name) = get_embed_model(
        embed_model_entry, auth_obj, retrieval=True
    )

    index, callback_manager, llama_debug = get_index(
        auth_obj, embed_model_entry, storage_dir, selected_names, names_tagged
    )

    llm_model_entry = get_llm_model_by_key(llm_model_key)

    prompt_id = (
        llm_model_entry["prompt"] if "prompt" in llm_model_entry else settings.PROMPT_ID
    )
    qa_prompt = QuestionAnswerPrompt(get_prompt(prompt_id))
    refine_prompt = RefinePrompt(get_prompt(settings.REFINE_PROMPT_ID))

    llm_service = llm_model_entry["service"]
    if "option" in llm_model_entry:
        if llm_model_entry["option"] == "CRAG":
            llm = get_llm_model(llm_model_entry, auth_obj)
            return LCBridge(index, embed_model, llm, auth_obj), llama_debug
    if llm_service == "langgraph":
        return LCBridge(index, embed_model, auth_obj), llama_debug

    llm = get_llm_model(llm_model_entry, auth_obj)
    langchain.llm_cache = SQLiteCache(database_path=settings.LANGCHAIN_LLM_CACHE_DB)

    service_context = ServiceContext.from_defaults(
        embed_model=embed_model, callback_manager=callback_manager, llm=llm
    )

    node_postprocessors = None
    similarity_top_k = settings.TOP_SIMIRALITY_K
    if "rerank" in llm_model_entry:
        from llama_index.legacy.postprocessor.cohere_rerank import CohereRerank

        cohere_rerank = CohereRerank(
            model=llm_model_entry["rerank"], top_n=settings.TOP_SIMIRALITY_K
        )
        node_postprocessors = [cohere_rerank]
        similarity_top_k = similarity_top_k + 2

    if "citation" in llm_model_entry:
        from llama_index.legacy.query_engine import CitationQueryEngine
        from llama_index.legacy.prompts import PromptTemplate

        index, callback_manager, llama_debug = get_index_with_llm(
            auth_obj, embed_model_entry, storage_dir, selected_names, names_tagged, llm
        )

        citation_prompt_id = (
            llm_model_entry["citation_prompt"]
            if "citation_prompt" in llm_model_entry
            else settings.CITATION_PROMPT_ID
        )
        citation_prompt = get_prompt(citation_prompt_id)
        query_engine = CitationQueryEngine.from_args(
            index,
            text_splitter=None,
            similarity_top_k=similarity_top_k,
            citation_qa_template=PromptTemplate(citation_prompt),
            node_postprocessors=node_postprocessors,
        )
    else:
        query_engine = index.as_query_engine(
            service_context=service_context,
            text_qa_template=qa_prompt,
            refine_template=refine_prompt,
            similarity_top_k=similarity_top_k,
            node_postprocessors=node_postprocessors,
        )

    return query_engine, llama_debug


def gen_embeddings_for_file(f, file_name, auth_obj, embed_key, tags="その他"):
    embed_model_entry = get_embed_model_by_key(embed_key)
    (embed_model, model_name) = get_embed_model(
        embed_model_entry, auth_obj, retrieval=False
    )
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])

    index_id = str(uuid.uuid4())

    service_context = ServiceContext.from_defaults(
        embed_model=embed_model, llm=None, callback_manager=callback_manager
    )

    documents = SimpleDirectoryReader(input_files=[f.name]).load_data()
    index = VectorStoreIndex.from_documents(
        documents=documents, service_context=service_context
    )
    index.storage_context.persist(persist_dir=os.path.join(MULTI_STORAGE_DIR, index_id))

    storage_record = {
        "path": index_id,
        "file_name": file_name,
        "embed_model": model_name,
        "tags": tags.split(","),
    }

    add_record_to_storage_config(MULTI_STORAGE_DIR, index_id, storage_record)

    return storage_record
