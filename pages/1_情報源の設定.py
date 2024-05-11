import streamlit as st

import json

from index_module import (
    MULTI_STORAGE_DIR,
    PREFIX_FOR_INDEXES,
    KEY_FOR_INDEXES,
    get_storage_config,
)


def get_config(storage_dir):
    return get_storage_config(storage_dir)


st.title("Settings")

# print("Before:", st.session_state)
## Index Selection
with st.container():
    config = get_config(MULTI_STORAGE_DIR)
    all_indexes = False
    selected_indexes = set()
    if KEY_FOR_INDEXES not in st.session_state:
        all_indexes = True
    else:
        for index in json.loads(st.session_state[KEY_FOR_INDEXES]):
            selected_indexes.add(index)

    # print(" - Selected", selected_indexes)
    selected_state = {}
    for storage_key in config:
        key = PREFIX_FOR_INDEXES + storage_key
        storage = config[storage_key]
        if all_indexes:
            selected_state[storage_key] = st.checkbox(
                storage["file_name"], value=True, key=key
            )
        elif key in st.session_state:
            # print("  - Key in ", key)
            selected_state[storage_key] = st.checkbox(
                storage["file_name"], value=st.session_state[key], key=key
            )
        else:
            selected_state[storage_key] = st.checkbox(
                storage["file_name"], value=(storage_key in selected_indexes), key=key
            )
    st.session_state[KEY_FOR_INDEXES] = json.dumps(
        [k for k, v in selected_state.items() if v]
    )

# print("After:", st.session_state)
