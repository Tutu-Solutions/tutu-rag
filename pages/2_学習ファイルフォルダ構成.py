import streamlit as st
import extra_streamlit_components as stx
from streamlit_tree_select import tree_select

import json

from index_module import (
    MULTI_STORAGE_DIR,
    PREFIX_FOR_INDEXES,
    KEY_FOR_INDEXES,
    get_storage_names,
    get_select_tree,
    STATE_KEY_FOR_CHECKED_NODES,
    STATE_KEY_FOR_EXPANDED_NODES,
    COOKIE_KEY_FOR_CHECKED_NODES,
    COOKIE_KEY_FOR_EXPANDED_NODES,
)

st.set_page_config(layout="wide")
st.title("学習ファイルフォルダ構成")

## Index Selection
cookie_manager = stx.CookieManager()

from datetime import datetime, timedelta

expire_time = datetime.now() + timedelta(weeks=4)

selected_names_cookie = cookie_manager.get(cookie="selected_names")
expanded_node_cookie = cookie_manager.get(cookie=STATE_KEY_FOR_EXPANDED_NODES)
checked_node_cookie = cookie_manager.get(cookie=STATE_KEY_FOR_CHECKED_NODES)

with st.container(border=True):
    (select_tree, tags, all_keys) = get_select_tree(MULTI_STORAGE_DIR)
    expanded = tags
    checked = all_keys

    if STATE_KEY_FOR_EXPANDED_NODES in st.session_state:
        expanded = st.session_state[STATE_KEY_FOR_EXPANDED_NODES]
    elif expanded_node_cookie:
        if expanded_node_cookie != "all":
            expanded = expanded_node_cookie

    if STATE_KEY_FOR_CHECKED_NODES in st.session_state:
        checked = st.session_state[STATE_KEY_FOR_CHECKED_NODES]
    elif checked_node_cookie:
        if checked_node_cookie != "all":
            checked = checked_node_cookie

    ret_select = tree_select(select_tree, expanded=expanded, checked=checked)

    cols = st.columns([1, 1, 7])
    with cols[0]:
        submitted = st.button("適用")
    with cols[1]:
        reset = st.button("リセット")

    if reset:
        with st.container(height=1, border=False):
            if STATE_KEY_FOR_EXPANDED_NODES in st.session_state:
                del st.session_state[STATE_KEY_FOR_EXPANDED_NODES]
            if STATE_KEY_FOR_CHECKED_NODES in st.session_state:
                del st.session_state[STATE_KEY_FOR_CHECKED_NODES]
            if expanded_node_cookie:
                cookie_manager.set(
                    STATE_KEY_FOR_EXPANDED_NODES,
                    "all",
                    key=COOKIE_KEY_FOR_EXPANDED_NODES,
                    expires_at=expire_time,
                )
            if checked_node_cookie:
                cookie_manager.set(
                    STATE_KEY_FOR_CHECKED_NODES,
                    "all",
                    key=COOKIE_KEY_FOR_CHECKED_NODES,
                    expires_at=expire_time,
                )

    if submitted:
        with st.container(height=1, border=False):
            st.session_state[STATE_KEY_FOR_EXPANDED_NODES] = ret_select["expanded"]
            st.session_state[STATE_KEY_FOR_CHECKED_NODES] = ret_select["checked"]
            cookie_manager.set(
                STATE_KEY_FOR_EXPANDED_NODES,
                json.dumps(ret_select["expanded"]),
                key=COOKIE_KEY_FOR_EXPANDED_NODES,
                expires_at=expire_time,
            )
            cookie_manager.set(
                STATE_KEY_FOR_CHECKED_NODES,
                json.dumps(ret_select["checked"]),
                key=COOKIE_KEY_FOR_CHECKED_NODES,
                expires_at=expire_time,
            )
        st.success("処理完了")
