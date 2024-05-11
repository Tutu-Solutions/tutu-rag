import streamlit as st
import extra_streamlit_components as stx

import os
import json
import openai

## Local Import
from index_module import get_multi_vector_index
from index_module import put_result_table
from index_module import KEY_FOR_INDEXES, MULTI_STORAGE_DIR

## LlamaIndex Import
from llama_index import ServiceContext
from llama_index.prompts.prompts import RefinePrompt
from llama_index.query_engine.multistep_query_engine import MultiStepQueryEngine
from llama_index.indices.query.query_transform.base import StepDecomposeQueryTransform
from llama_index.llms import Gemini

## Multi Step Query Import
from llama_index.prompts.prompts import QuestionAnswerPrompt
from llama_index import LLMPredictor

## LlamaIndex Debug
from llama_index.callbacks import CallbackManager, LlamaDebugHandler, CBEventType

## LLM Models
from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

## LLM Cache
import langchain
from langchain.cache import SQLiteCache

# Local Embedding
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings

## GLOBAL
TOP_SIMIRALITY_K = 3
OPTION_TO_MODEL={
        "全て" : "all",
        "GPT-3.5" : "gpt-3.5-turbo",
        "GPT-4" : "gpt-4-turbo-preview",
        "Gemini" : "gemini-ultra"
        }
OPENAI_EMBED_MODEL="text-embedding-3-large"
#LOCAL_EMBED_MODEL="intfloat/multilingual-e5-large"
LOCAL_EMBED_MODEL="intfloat/multilingual-e5-small"

@st.cache_resource
def get_embed_model(api_key, use_local_embed, use_gemini_embed):
    from langchain.storage import LocalFileStore
    from langchain.embeddings import CacheBackedEmbeddings
    from langchain_openai import OpenAIEmbeddings
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    
    embed_store = LocalFileStore("./embed_cache/")
    
    if use_local_embed:
        model_name = LOCAL_EMBED_MODEL
        underlying_embed_model = HuggingFaceEmbeddings(model_name=model_name)
    elif use_gemini_embed:
        model_name = GEMINI_EMBED_MODEL
        underlying_embed_model = GoogleGenerativeAIEmbeddings(model=model_name)
    else:
        model_name = OPENAI_EMBED_MODEL
        underlying_embed_model = OpenAIEmbeddings(model=model_name)

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
                underlying_embed_model, embed_store, namespace=model_name
                )

    return (LangchainEmbedding(cached_embedder), model_name)

@st.cache_resource
def get_query_engines(api_key, g_api_key, model, storage_dir, selected_indexes_ids, use_local_embed=True):
    langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    use_gemini_embed = False
    if model.startswith("gemini"):
        os.environ["GOOGLE_API_KEY"] = g_api_key
        llm = Gemini(model="models/"+model)
        use_gemini_embed = True
        # Need update langchain >0.0.350
        #llm = ChatGoogleGenerativeAI(model=model)
    else:
        os.environ["OPENAI_API_KEY"] = api_key
        llm = ChatOpenAI(model_name=model)

    (embed_model, embed_model_name) = get_embed_model(api_key, use_local_embed, use_gemini_embed)
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
        callback_manager=callback_manager, llm=llm
    )

    index = get_multi_vector_index(selected_indexes_ids, storage_dir, service_context, embed_model_name)

    QA_PROMPT_TMPL = (
        "以下の情報を参照してください。 \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "この情報を使って、次の質問に日本語で答えてください。: {query_str}\n"
    )
    qa_prompt = QuestionAnswerPrompt(QA_PROMPT_TMPL)
    REFINE_PROMPT = (
        "元の質問は次のとおりです: {query_str} \n"
        "既存の回答を提供しました: {existing_answer} \n"
        "既存の答えを洗練する機会があります \n"
        "(必要な場合のみ)以下にコンテキストを追加します。 \n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "新しいコンテキストを考慮して、元の答えをより良く洗練して質問に答えてください。\n"
        "コンテキストが役に立たない場合は、元の回答と同じものを返します。"
    )
    refine_prompt = RefinePrompt(REFINE_PROMPT)

    base_query_engine = index.as_query_engine(
        service_context=service_context,
        text_qa_template=qa_prompt,
        refine_template=refine_prompt,
        similarity_top_k=TOP_SIMIRALITY_K,
    )

    return base_query_engine, llama_debug


def main_chat():
    st.set_page_config(layout="wide")

    cookie_manager = stx.CookieManager()
    api_key = cookie_manager.get(cookie="api_key")
    model = cookie_manager.get(cookie="model")

    if api_key:
        api_key = st.text_input("OpenAI API Key", api_key, type="password")
    else:
        api_key = st.text_input("OpenAI API Key", type="password")

    g_api_key = st.text_input("Google API Key", type="password")
    
    with st.form("question"):
        question = st.text_input("Your question", max_chars=1024)
        col1, col2 = st.columns(2)
        with col1:
            st.form_submit_button("質問") 

        with col2:
                st.radio(
                    "LLMを選んでください。",
                    key="model",
                    options=OPTION_TO_MODEL.keys(),
                )
    if question:
        result_area = st.empty()
        info_area = st.empty()
        
        if "model" not in st.session_state:
            st.session_state.model = "GPT-3.5"
        model = OPTION_TO_MODEL[st.session_state.model]

        if KEY_FOR_INDEXES not in st.session_state:
            selected_indexes_ids = "all"
        else:
            selected_indexes_ids = json.loads(st.session_state[KEY_FOR_INDEXES])
        
        
        cookie_manager.set("api_key", api_key)
        
        try:
            if model == "all":
                results = []
                for model_key in OPTION_TO_MODEL:
                    llm_model = OPTION_TO_MODEL[model_key]
                    if llm_model == "all":
                        continue
                    query_engine, llama_debug = get_query_engines(api_key, g_api_key, llm_model, MULTI_STORAGE_DIR, selected_indexes_ids)

                    try:
                        res = query_engine.query(question)
                        event_pairs = llama_debug.get_llm_inputs_outputs()
                        llama_debug.flush_event_logs()
                        prompt =  event_pairs[-1][1].payload["messages"][0]
                        results.append((model_key, question,prompt,res,llm_model))
                    except Exception as e:
                        res = {"response" : e.message}
                        results.append((model_key, question, None, res, llm_model))

                cols = st.columns(len(OPTION_TO_MODEL) - 1)
                i = 0
                for (model_key, question, prompt, res, model) in results:
                    with cols[i]:
                        st.subheader(model_key)
                        st.write(res.response)

                    i = i + 1

                for (model_key, question, prompt, res, model) in results:
                    if prompt:
                        put_result_table(question, prompt, res, model)
            else:
                base_query_engine, llama_debug = get_query_engines(
                    api_key, g_api_key, model, MULTI_STORAGE_DIR, selected_indexes_ids
                )
                res = base_query_engine.query(question)
                event_pairs = llama_debug.get_llm_inputs_outputs()
                llama_debug.flush_event_logs()
                prompt = event_pairs[-1][1].payload["messages"][0]
                put_result_table(question, prompt,res, model)

                print(res.response)
                result_area.write(res.response)
        except Exception as e:
            import traceback
            print(e)
            traceback.print_exc()
            st.warning("例外が発生しました。")
            st.text(e)


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)4s %(name)s: %(message)s",
    filename="%s.log" % (os.path.basename(__file__)),
)

main_chat()
