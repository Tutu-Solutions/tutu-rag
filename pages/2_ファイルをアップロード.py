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


@st.cache_resource
def setup(api_key):
    openai.api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key


def main():
    st.title("File Upload")
    uploaded_file = st.file_uploader("Upload Here")

    cookie_manager = stx.CookieManager()
    api_key = cookie_manager.get(cookie="api_key")

    if api_key:
        api_key = st.text_input("OpenAI API Key", api_key, type="password")
    else:
        api_key = st.text_input("OpenAI API Key")

    if uploaded_file:
        if not api_key:
            st.warning("Enter your API Key")
            return
        setup(api_key)

        file_name = uploaded_file.name
        bytes_data = uploaded_file.getvalue()
        with NamedTemporaryFile(mode="wb") as f:
            with f.file as temp_file:
                f.write(bytes_data)

            index_id = str(uuid.uuid4())

            llama_debug = LlamaDebugHandler(print_trace_on_end=True)
            callback_manager = CallbackManager([llama_debug])
            service_context = ServiceContext.from_defaults(
                callback_manager=callback_manager
            )

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
            }

            add_record_to_storage_config(MULTI_STORAGE_DIR, index_id, storage_record)


main()
