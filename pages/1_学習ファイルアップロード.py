import streamlit as st
import extra_streamlit_components as stx

import os

from tempfile import NamedTemporaryFile

## Local Import
import settings
from engine import AuthRecord
from engine import get_embed_model_keys
from engine import gen_embeddings_for_file

## GLOBAL
DEBUG = False

st.set_page_config(layout="wide")
st.title("学習ファイルアップロード")

cookie_manager = stx.CookieManager()
api_key = cookie_manager.get(cookie="api_key")
g_api_key = cookie_manager.get(cookie="g_api_key")

with st.expander("API Key", expanded=api_key == None):
    if api_key:
        api_key = st.text_input("OpenAI API Key", api_key, type="password")
    else:
        api_key = st.text_input("OpenAI API Key", type="password")
    if g_api_key:
        g_api_key = st.text_input("Google API Key", g_api_key, type="password")
    else:
        g_api_key = st.text_input("Google API Key", type="password")

with st.form("file_upload"):
    tags = st.text_input("フォルダ名")
    uploaded_file = st.file_uploader("Upload Here", label_visibility="collapsed")
    submitted = st.form_submit_button("アップロード")

if submitted:
    process = True
    if uploaded_file:
        if not api_key:
            st.warning("OpenAI API Keyを入力してください。")
            process = False
        if not g_api_key:
            st.warning("Google API Keyを入力してください。")
            process = False
        if not tags:
            st.warning("フォルダ名を入力してください。")
            process = False
    else:
        process = False

    #    with st.container(height=1, border=False):
    #        from datetime import datetime, timedelta
    #        expire_time = datetime.now() + timedelta(weeks=4)
    #
    #        cookie_manager.set(
    #            "api_key", api_key, key="api_key_cookie", expires_at=expire_time
    #        )
    #        cookie_manager.set(
    #            "g_api_key", g_api_key, key="g_api_key_cookie", expires_at=expire_time
    #        )

    if process:
        auth_obj = AuthRecord(api_key, "", "", "", "")

        file_name = uploaded_file.name
        bytes_data = uploaded_file.getvalue()
        with NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(bytes_data)
            f.close()

            with open("uploaded.list", "a", encoding="utf-8") as l:
                l.write("%s,%s\n" % (file_name, f.name))
                l.close()

            progress_bar = st.progress(0, text="処理待ち")

            progress = 0
            embed_model_keys = get_embed_model_keys()
            counts = len(embed_model_keys)
            for embed_key in embed_model_keys:
                rec = gen_embeddings_for_file(f, file_name, auth_obj, embed_key, tags)
                # if settings.DEBUG_MODE:
                #    st.write(rec)
                progress = progress + 1
                print(progress)
                progress_bar.progress(
                    int(progress / counts * 100),
                    text="埋め込み %d/%d 完成しました" % (progress, counts),
                )

            progress_bar.progress(100, text="すべての埋め込み完成しました")
            st.success("処理完了")
