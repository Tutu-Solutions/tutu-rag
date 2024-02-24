import streamlit as st

import os
import json
import uuid

## LlamaIndex Import
from llama_index import ServiceContext, StorageContext
from llama_index import SimpleDirectoryReader
from llama_index.indices.vector_store import VectorStoreIndex

## LlamaIndex Debug
from llama_index.callbacks import CallbackManager, LlamaDebugHandler, CBEventType

## Embeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings import VertexAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

## Prompts Import
from llama_index.prompts.prompts import QuestionAnswerPrompt
from llama_index.prompts.prompts import RefinePrompt

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

## GLOBAL
LLM_CONFIG_FILE = "llm_model.json"

OPTION_TO_EMBED_MODEL = {
    #        "自動": {"service":"auto"},
    EMBED_KEY_OPENAI: {"service": "openai", "model": "text-embedding-ada-002"},
    #        "Gemini" : {"service":"google", "model":"models/embedding-001"},
    EMBED_KEY_ONPREMISE: {
        "service": "local",
        "model": "intfloat/multilingual-e5-small",
    },
    #        "オンプレ large" : {"service":"local", "model":"intfloat/multilingual-e5-large"}
}


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
    with open(LLM_CONFIG_FILE, encoding="utf-8") as f:
        return json.load(f)


def get_llm_model_categories():
    config = get_llm_model_config()
    categories = set()
    for k in config:
        if "hidden" in config[k] and config[k]["hidden"] == "True":
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
        if "hidden" in config[k] and config[k]["hidden"] == "True":
            continue
        if config[k]["category"] == category:
            ret.append(k)
    return ret


@st.cache_resource
def get_embed_model_keys():
    return OPTION_TO_EMBED_MODEL.keys()


@st.cache_resource
def get_embed_model(embed_model):
    embed_store = LocalFileStore("./embed_cache/")

    service = embed_model["service"]
    model_name = embed_model["model"]

    if service == "local":
        underlying_embed_model = HuggingFaceEmbeddings(model_name=model_name)
    elif service == "google":
        underlying_embed_model = GoogleGenerativeAIEmbeddings(model=model_name)
    elif service == "openai":
        underlying_embed_model = OpenAIEmbeddings(model=model_name)

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embed_model, embed_store, namespace=model_name.replace("@", "_")
    )

    return (LangchainEmbedding(cached_embedder), model_name)


@st.cache_resource(hash_funcs={AuthRecord: lambda x: str(str.__dict__)})
def get_llm_model(llm_model_entry, auth_obj):
    llm_service = llm_model_entry["service"]
    llm_model = llm_model_entry["model"]
    if llm_service == "Google":
        return GoogleGenerativeAI(model=llm_model, google_api_key=auth_obj.g_api_key)
    elif llm_service == "Anthropic":
        return ChatAnthropic(model=llm_model, anthropic_api_key=auth_obj.anthropic_key)
    elif llm_service == "Amazon":
        return BedrockChat(client=auth_obj.bedrock_runtime, model_id=llm_model)
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
        return ChatOpenAI(model_name=llm_model)


def get_prompt(prompt_id):
    import yaml
    from pathlib import Path

    prompt_dict = yaml.safe_load(Path("prompts.yml").read_text())
    prompt = "\n".join(prompt_dict[prompt_id])
    if settings.DEBUG_MODE:
        print(prompt)
    return prompt


@st.cache_resource
def get_index(
    embed_model_key,
    storage_dir,
    selected_names,
    names_tagged,
):
    # TODO: Pass in Callback Manager from top level
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])

    embed_model = OPTION_TO_EMBED_MODEL[embed_model_key]
    (embed_model, embed_model_name) = get_embed_model(embed_model)

    service_context = ServiceContext.from_defaults(
        embed_model=embed_model, callback_manager=callback_manager, llm=None
    )

    return (
        get_multi_vector_index(
            selected_names, storage_dir, service_context, embed_model_name, names_tagged
        ),
        callback_manager,
        llama_debug,
    )


@st.cache_resource(hash_funcs={AuthRecord: lambda x: str(str.__dict__)})
def get_query_engine(
    auth_obj,
    llm_model_key,
    embed_model_key,
    storage_dir,
    selected_names,
    names_tagged=False,
):
    llm_model_entry = get_llm_model_by_key(llm_model_key)
    llm = get_llm_model(llm_model_entry, auth_obj)
    langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

    index, callback_manager, llama_debug = get_index(
        embed_model_key, storage_dir, selected_names, names_tagged
    )

    embed_model = OPTION_TO_EMBED_MODEL[embed_model_key]
    (embed_model, embed_model_name) = get_embed_model(embed_model)

    prompt_id = (
        llm_model_entry["prompt"] if "prompt" in llm_model_entry else settings.PROMPT_ID
    )
    qa_prompt = QuestionAnswerPrompt(get_prompt(prompt_id))
    refine_prompt = RefinePrompt(get_prompt(settings.REFINE_PROMPT_ID))

    service_context = ServiceContext.from_defaults(
        embed_model=embed_model, callback_manager=callback_manager, llm=llm
    )

    base_query_engine = index.as_query_engine(
        service_context=service_context,
        text_qa_template=qa_prompt,
        refine_template=refine_prompt,
        similarity_top_k=settings.TOP_SIMIRALITY_K,
    )

    return base_query_engine, llama_debug


def gen_embeddings_for_file(f, file_name, embed_key, tags="その他"):
    embed_model_entry = OPTION_TO_EMBED_MODEL[embed_key]
    (embed_model, model_name) = get_embed_model(embed_model_entry)
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
