import streamlit as st
import extra_streamlit_components as stx

import os
import time
import json
import argparse

## Local Import
import settings

from index_module import put_result_table
from index_module import MULTI_STORAGE_DIR
from index_module import PREFIX_FOR_LLM, KEY_FOR_SELECTED_LLMS
from index_module import STATE_KEY_FOR_CHECKED_NODES
from engine import get_embed_model_keys
from engine import get_query_engine
from engine import get_llm_model_categories, get_llm_model_by_categories
from engine import get_llm_model_by_key
from engine import AuthRecord
from keys import *

## Global
IBM_ENDPOINT_LIST = [
    "東京,https://jp-tok.ml.cloud.ibm.com",
    "ダラス,https://private.us-south.ml.cloud.ibm.com",
    "ロンドン,https://private.eu-gb.ml.cloud.ibm.com",
    "フランクフルト,https://private.eu-de.ml.cloud.ibm.com",
]

AWS_REGION_LIST = [
    "米国西部（オレゴン）,us-west-2",
    "米国東部（バージニア) ,us-east-1",
    "アジアパシフィック（東京）,ap-northeast-1",
]


def run_query(
    auth_obj,
    question,
    llm_model_key,
    embed_model_key,
    storage_dir,
    selected_names,
):
    print(selected_names)
    llm_model = get_llm_model_by_key(llm_model_key)["model"]
    query_engine, llama_debug = get_query_engine(
        auth_obj,
        llm_model_key,
        embed_model_key,
        MULTI_STORAGE_DIR,
        selected_names,
        True,
    )

    dur = -1
    try:
        start_t = time.perf_counter()
        res = query_engine.query(question)
        end_t = time.perf_counter()
        dur = end_t - start_t
        event_pairs = llama_debug.get_llm_inputs_outputs()
        llama_debug.flush_event_logs()
        try:
            prompt = event_pairs[-1][1].payload["messages"][0]
        except Exception as e:
            print(e)
            try:
                ## For Text Completion Model
                prompt = event_pairs[-1][1].payload["formatted_prompt"]
            except Exception as e:
                print(e)
                prompt = ""
        return (llm_model_key, question, prompt, res, llm_model, dur)
    except Exception as e:

        class FakeRes(object):
            pass

        import traceback

        print(e)
        traceback.print_exc()
        res = FakeRes()
        res.response = str(e)
        return (llm_model_key, question, None, res, llm_model, dur)


def main_chat():
    st.set_page_config(layout="wide")

    cookie_manager = stx.CookieManager()
    api_key = cookie_manager.get(cookie="api_key")
    g_api_key = cookie_manager.get(cookie="g_api_key")
    anthropic_key = cookie_manager.get(cookie=STATE_KEY_FOR_ANTHROPIC_KEY)
    aws_access_key_id = cookie_manager.get(cookie=STATE_KEY_FOR_AWS_ACCESS_KEY)
    aws_secret_access_key = cookie_manager.get(cookie=STATE_KEY_FOR_AWS_SECRET_KEY)
    aws_region_name = cookie_manager.get(cookie=STATE_KEY_FOR_AWS_REGION)
    i_api_key = cookie_manager.get(cookie=STATE_KEY_FOR_IBM_KEY)
    i_project_id = cookie_manager.get(cookie=STATE_KEY_FOR_IBM_PROJECT_ID)
    i_auth_endpoint = cookie_manager.get(cookie=STATE_KEY_FOR_IBM_AUTH_ENDPOINT)
    selected_names = cookie_manager.get(cookie="selected_names")
    tagged_selected_names = cookie_manager.get(cookie=STATE_KEY_FOR_CHECKED_NODES)
    embed_model = cookie_manager.get(cookie=STATE_KEY_FOR_EMBED_MODEL)
    selected_llms_cookie = cookie_manager.get(cookie="selected_llms")

    with st.form("question"):
        key_filled = api_key != None and g_api_key != None and anthropic_key != None
        with st.expander("API Key", expanded=not key_filled):
            if api_key:
                api_key = st.text_input("OpenAI API Key", api_key, type="password")
            else:
                api_key = st.text_input("OpenAI API Key", type="password")

            if g_api_key:
                g_api_key = st.text_input("Google API Key", g_api_key, type="password")
            else:
                g_api_key = st.text_input("Google API Key", type="password")

            if anthropic_key:
                anthropic_key = st.text_input(
                    "Anthropic API Key", anthropic_key, type="password"
                )
            else:
                anthropic_key = st.text_input("Anthropic API Key", type="password")

            with st.container(border=False):
                cols = st.columns(3)
                with cols[0]:
                    if aws_access_key_id:
                        aws_access_key_id = st.text_input(
                            "AWS アクセスキー", aws_access_key_id, type="password"
                        )
                    else:
                        aws_access_key_id = st.text_input("AWS アクセスキー", type="password")
                with cols[1]:
                    if aws_secret_access_key:
                        aws_secret_access_key = st.text_input(
                            "AWS シークレットアクセスキー", aws_secret_access_key, type="password"
                        )
                    else:
                        aws_secret_access_key = st.text_input(
                            "AWS シークレットアクセスキー", type="password"
                        )
                with cols[2]:
                    if aws_region_name:
                        try:
                            index = AWS_REGION_LIST.index(aws_region_name)
                        except ValueError:
                            index = 0
                        aws_region_name = st.selectbox(
                            "AWS リージョン", AWS_REGION_LIST, index=index
                        )
                    else:
                        aws_region_name = st.selectbox("AWS リージョン", AWS_REGION_LIST)

            with st.container(border=False):
                cols = st.columns(3)
                with cols[0]:
                    if i_api_key:
                        i_api_key = st.text_input(
                            "IBM API Key", i_api_key, type="password"
                        )
                    else:
                        i_api_key = st.text_input("IBM API Key", type="password")
                with cols[1]:
                    if i_project_id:
                        i_project_id = st.text_input("IBM プロジェクトID", i_project_id)
                    else:
                        i_project_id = st.text_input("IBM プロジェクトID")
                with cols[2]:
                    if i_auth_endpoint:
                        try:
                            index = IBM_ENDPOINT_LIST.index(i_auth_endpoint)
                        except ValueError:
                            index = 0
                        i_auth_endpoint = st.selectbox(
                            "IBM エンドポイント", IBM_ENDPOINT_LIST, index=index
                        )
                    else:
                        i_auth_endpoint = st.selectbox("IBM エンドポイント", IBM_ENDPOINT_LIST)

        embed_model_keys = get_embed_model_keys()

        selected_llms = []
        all_llms = True
        if KEY_FOR_SELECTED_LLMS in st.session_state:
            all_llms = False
            for llm_model_key in json.loads(st.session_state[KEY_FOR_SELECTED_LLMS]):
                selected_llms.append(llm_model_key)
        elif selected_llms_cookie:
            selected_llms = selected_llms_cookie
            all_llms = False

        selected_llms_state = {}

        with st.expander("大規模言語モデル"):
            llm_categories = settings.PREDETERMINED_CATEGORIES
            llm_categories = llm_categories + [
                k for k in get_llm_model_categories() if k not in llm_categories
            ]
            cols = st.columns(settings.NUM_OF_COL_FOR_LLM)
            i = 0

            for category in llm_categories:
                with cols[i]:
                    st.text(category)
                    llm_models = get_llm_model_by_categories(category)
                    for llm in llm_models:
                        key = STATE_KEY_FOR_LLM_MODEL + llm.replace(" ", "_")
                        if all_llms:
                            selected_llms_state[llm] = st.checkbox(
                                llm, value=True, key=key
                            )
                        else:
                            selected_llms_state[llm] = st.checkbox(
                                llm, value=(llm in selected_llms), key=key
                            )
                i = (i + 1) % settings.NUM_OF_COL_FOR_LLM

        question = st.text_input("質問", max_chars=1024)
        col1, col2 = st.columns([1, 5])
        with col1:
            st.form_submit_button("送信")

        if embed_model:
            try:
                selected_embed_index = list(embed_model_keys).index(embed_model)
            except ValueError:
                selected_embed_index = 0
        else:
            selected_embed_index = 0

        with col2:
            col1, col2 = st.columns([1, 5])
            with col1:
                st.text("ベクトル化方法")
            with col2:
                st.radio(
                    "ベクトル化方法",
                    key="embed_model",
                    index=selected_embed_index,
                    options=embed_model_keys,
                    horizontal=True,
                    label_visibility="collapsed",
                )

    if question:
        if not selected_names:
            selected_names = "all"
        if not tagged_selected_names:
            tagged_selected_names = "all"

        selected_llm_model_keys = [k for k, v in selected_llms_state.items() if v]
        selected_llms = json.dumps(selected_llm_model_keys)
        st.session_state[KEY_FOR_SELECTED_LLMS] = selected_llms

        embed_model_key = st.session_state.embed_model

        with st.container(height=1, border=False):
            cookie_manager.set("api_key", api_key, key="api_key_cookie")
            cookie_manager.set("g_api_key", g_api_key, key="g_api_key_cookie")
            cookie_manager.set(
                STATE_KEY_FOR_EMBED_MODEL,
                embed_model_key,
                key=COOKIE_KEY_FOR_EMBED_MODEL,
            )
            cookie_manager.set(
                "selected_llms", selected_llms, key="selected_llms_cookie"
            )
            cookie_manager.set(
                STATE_KEY_FOR_IBM_KEY, i_api_key, key=COOKIE_KEY_FOR_IBM_KEY
            )
            cookie_manager.set(
                STATE_KEY_FOR_IBM_PROJECT_ID,
                i_project_id,
                key=COOKIE_KEY_FOR_IBM_PROJECT_ID,
            )
            cookie_manager.set(
                STATE_KEY_FOR_IBM_AUTH_ENDPOINT,
                i_auth_endpoint,
                key=COOKIE_KEY_FOR_IBM_AUTH_ENDPOINT,
            )
            cookie_manager.set(
                STATE_KEY_FOR_ANTHROPIC_KEY,
                anthropic_key,
                key=COOKIE_KEY_FOR_ANTHROPIC_KEY,
            )
            cookie_manager.set(
                STATE_KEY_FOR_AWS_ACCESS_KEY,
                aws_access_key_id,
                key=COOKIE_KEY_FOR_AWS_ACCESS_KEY,
            )
            cookie_manager.set(
                STATE_KEY_FOR_AWS_SECRET_KEY,
                aws_secret_access_key,
                key=COOKIE_KEY_FOR_AWS_SECRET_KEY,
            )
            cookie_manager.set(
                STATE_KEY_FOR_AWS_REGION, aws_region_name, key=COOKIE_KEY_FOR_AWS_REGION
            )

        auth_obj = AuthRecord(
            api_key, g_api_key, i_api_key, i_project_id, i_auth_endpoint
        )
        auth_obj.add_anthropic_key(anthropic_key)
        auth_obj.add_aws_key(aws_access_key_id, aws_secret_access_key, aws_region_name)

        os.environ["OPENAI_API_KEY"] = api_key

        try:
            results = []
            for model_key in selected_llm_model_keys:
                result = run_query(
                    auth_obj,
                    question,
                    model_key,
                    embed_model_key,
                    MULTI_STORAGE_DIR,
                    tagged_selected_names,
                )

                results.append(result)

        except Exception as e:
            import traceback

            print(e)
            traceback.print_exc()
            st.warning("例外が発生しました。")
            st.write(e)

        i = 0
        print("+++ +++ +++ +++ +++")
        print(question)
        print("+++ +++ +++ +++ +++")
        for model_key, question, prompt, res, model, dur in results:
            if i == 0:
                container = st.container()
                with container:
                    result_cols = st.columns(3)
                    st.divider()
            with result_cols[i]:
                st.subheader("%s (%.2fs)" % (model_key, dur))
                st.write(res.response.replace("\n", "  \n"))
                print(" - %s, 処理時間: %.2fs" % (model_key, dur))
                print(res.response)
                print("--- --- --- --- ---")

            i = i + 1
            if i == 3:
                i = 0

        for model_key, question, prompt, res, model, dur in results:
            if prompt:
                put_result_table(question, prompt, res, model)


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)4s %(name)s: %(message)s",
    filename="%s.log" % (os.path.basename(__file__)),
)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true")
parser.add_argument("-k", "--top_k",  type=int, default=settings.TOP_SIMIRALITY_K)
parser.add_argument("-p", "--prompt", default=settings.PROMPT_ID)

args = parser.parse_args()

settings.DEBUG_MODE = args.debug
settings.TOP_SIMIRALITY_K = args.top_k
settings.PROMPT_ID = args.prompt

main_chat()
