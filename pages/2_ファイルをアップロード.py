import streamlit as st
import extra_streamlit_components as stx

import os
import uuid
import openai

from tempfile import NamedTemporaryFile

## Local Import
from index_module import MULTI_STORAGE_DIR
from index_module import add_record_to_storage_config

## LlamaIndex Import
from llama_index import ServiceContext, StorageContext

from llama_index import SimpleDirectoryReader
from llama_index.indices.vector_store import VectorStoreIndex

## LlamaIndex Debug
from llama_index.callbacks import CallbackManager, LlamaDebugHandler, CBEventType

from llama_index.embeddings.langchain import LangchainEmbedding

OPTION_TO_EMBED_MODEL={
        "OpenAI" : {"service":"openai", "model": "text-embedding-ada-002"},
        "オンプレ(e5-small)" : {"service":"local", "model":"intfloat/multilingual-e5-small"}
        }

def gen_embeddings(f, file_name, embed_model_entry):
    from langchain.storage import LocalFileStore
    from langchain.embeddings import CacheBackedEmbeddings
    from langchain_openai import OpenAIEmbeddings
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain.embeddings import VertexAIEmbeddings
    from langchain.embeddings import HuggingFaceEmbeddings

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    
    embed_store = LocalFileStore("./embed_cache/")

    model_name = embed_model_entry["model"]
    if embed_model_entry["service"] == "openai":
        underlying_embed_model = OpenAIEmbeddings(model=model_name)
    elif embed_model_entry["service"] == "local":
        underlying_embed_model = HuggingFaceEmbeddings(model_name=model_name)

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(underlying_embed_model, embed_store, namespace=model_name.replace("@","_"))

    index_id = str(uuid.uuid4())

    service_context = ServiceContext.from_defaults(embed_model=LangchainEmbedding(cached_embedder), llm=None, callback_manager=callback_manager)

    documents = SimpleDirectoryReader(input_files=[f.name]).load_data()
    index = VectorStoreIndex.from_documents(
        documents=documents, service_context=service_context
    )
    index.storage_context.persist(
        persist_dir=os.path.join(MULTI_STORAGE_DIR, index_id)
    )

    storage_record = {
        "path": index_id,
        "file_name": file_name,
        "embed_model" : model_name,
    }

    add_record_to_storage_config(MULTI_STORAGE_DIR, index_id, storage_record)

    return storage_record

@st.cache_resource
def setup(api_key):
    openai.api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key


def main():
    st.set_page_config(layout="wide")
    st.title("ファイルをアップロード")
    uploaded_file = st.file_uploader("Upload Here")

    cookie_manager = stx.CookieManager()
    api_key = cookie_manager.get(cookie="api_key")

    with st.expander("API Key", expanded=api_key==None):
        if api_key:
            api_key = st.text_input("OpenAI API Key", api_key, type="password")
        else:
            api_key = st.text_input("OpenAI API Key", type="password")

    if uploaded_file:
        if not api_key:
            st.warning("Enter your API Key")
            return

        with st.empty():
            #cookie_manager.set("api_key", api_key, key="api_key")
            setup(api_key)

        file_name = uploaded_file.name
        bytes_data = uploaded_file.getvalue()
        with NamedTemporaryFile(mode="wb") as f:
            with f.file as temp_file:
                f.write(bytes_data)

            progress_bar = st.progress(0, text="処理待ち")
#            step = int(100 / len(OPTION_TO_EMBED_MODEL.keys()))

            progress = 0
            counts = len(OPTION_TO_EMBED_MODEL.keys())
            for embed_key in OPTION_TO_EMBED_MODEL.keys():
                rec = gen_embeddings(f, file_name, OPTION_TO_EMBED_MODEL[embed_key])
                progress = progress + 1
                progress_bar.progress(int(progress / counts * 100), text="埋め込み %d/%d 完成しました" % (progress, counts))
                #st.write(rec)

            progress_bar.progress(100, text="すべての埋め込み完成しました")
            st.success("処理完了")

            #index_id = str(uuid.uuid4())

            #llama_debug = LlamaDebugHandler(print_trace_on_end=True)
            #callback_manager = CallbackManager([llama_debug])
            #service_context = ServiceContext.from_defaults(
            #    callback_manager=callback_manager
            #)

            #documents = SimpleDirectoryReader(input_files=[f.name]).load_data()
            #index = VectorStoreIndex.from_documents(
            #    documents=documents, service_context=service_context
            #)
            #index.storage_context.persist(
            #    persist_dir=os.path.join(MULTI_STORAGE_DIR, index_id)
            #)

            #storage_record = {
            #    "path": index_id,
            #    "file_name": file_name,
            #}

            #add_record_to_storage_config(MULTI_STORAGE_DIR, index_id, storage_record)


main()
