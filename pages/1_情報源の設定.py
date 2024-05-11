import streamlit as st
import extra_streamlit_components as stx

import json

from index_module import (
    MULTI_STORAGE_DIR,
    PREFIX_FOR_INDEXES,
    KEY_FOR_INDEXES,
    get_storage_names,
)

st.set_page_config(layout="wide")
st.title("情報源の設定")

## Index Selection
cookie_manager = stx.CookieManager()

selected_names_cookie = cookie_manager.get(cookie="selected_names")
with st.form('index_select'):
    all_indexes = False

    selected_names = []
    if KEY_FOR_INDEXES not in st.session_state:
        if selected_names_cookie:
            if selected_names_cookie == "all":
                all_indexes = True
            else:
                selected_names = selected_names_cookie
        else:
            all_indexes = True
    else:
        for name in json.loads(st.session_state[KEY_FOR_INDEXES]):
            selected_names.append(name)

    selected_state = {}
    names = get_storage_names(MULTI_STORAGE_DIR)
    for name in names:
        key = PREFIX_FOR_INDEXES + name
        if all_indexes:
            selected_state[name] = st.checkbox(
                name, value=True, key=key
            )
        else:
            selected_state[name] = st.checkbox(
                name, value=(name in selected_names), key=key
            )

    submitted = st.form_submit_button("適用")

if submitted:
    selected_names = json.dumps(
        [k for k, v in selected_state.items() if v],
        ensure_ascii=False
    )
    st.session_state[KEY_FOR_INDEXES] = selected_names
    cookie_manager.set("selected_names", selected_names, key="selected_names")
    selected_names_cookie = cookie_manager.get(cookie="selected_names")

