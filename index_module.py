import streamlit as st

import os
import json

## LlamaIndex Import
from llama_index import StorageContext, load_index_from_storage

## Local Import
import settings
from multi_vector_index import MultiVectorStore

## Keyword Definitions
MULTI_STORAGE_DIR = "./multi_storage"
CONFIG_FILE = "config.json"

ATTRIBUTE_FOR_PATH = "path"
ATTRIBUTE_FOR_EMBED_MODEL = "embed_model"

PREFIX_FOR_INDEXES = "index_"
PREFIX_FOR_LLM = "llms_"
KEY_FOR_SELECTED_LLMS = "selected_llms"
KEY_FOR_INDEXES = "indexes"
KEY_FOR_TAGS = "tags"
KEY_FOR_NAME = "file_name"

STATE_KEY_FOR_CHECKED_NODES = "state_checked"
STATE_KEY_FOR_EXPANDED_NODES = "state_expanded"

COOKIE_KEY_FOR_CHECKED_NODES = "cookie_checked"
COOKIE_KEY_FOR_EXPANDED_NODES = "cookie_expanded"

# RECORD_URL="http://127.0.0.1:5001/gpt-dev-34c73/us-central1/add"
RECORD_URL = "https://asia-northeast1-gpt-dev-34c73.cloudfunctions.net/add_result"


def get_storage_config(storage_dir):
    with open(os.path.join(storage_dir, CONFIG_FILE), encoding="utf-8") as f:
        config = json.load(f)
        return config["storages"]


def add_record_to_storage_config(storage_dir, index_id, storage_record):
    with open(os.path.join(storage_dir, CONFIG_FILE), encoding="utf-8") as f:
        config = json.load(f)
        config["storages"][index_id] = storage_record

    json_str = json.dumps(config, indent=4, ensure_ascii=False)

    with open(os.path.join(storage_dir, CONFIG_FILE), "w", encoding="utf-8") as f:
        f.write(json_str)


def get_storage_names(storage_dir):
    config = get_storage_config(storage_dir)
    names = set()
    for storage_key in config:
        names.add(config[storage_key]["file_name"])
    return sorted(names)


def get_tagged_name(tag, name):
    return tag + "+" + name


def get_storage_keys_from_names(config, names, names_tagged=False):
    names_set = set(names)
    selected_indexes_ids = set()
    for storage_key in config.keys():
        fname = config[storage_key]["file_name"]
        if not names_tagged:
            if fname in names_set:
                selected_indexes_ids.add(storage_key)
        elif "tags" in config[storage_key]:
            for tag in config[storage_key]["tags"]:
                name = get_tagged_name(tag, fname)
                if name in names_set:
                    selected_indexes_ids.add(storage_key)
    return list(selected_indexes_ids)


def get_tag_names(config):
    tags = set()
    for key in config:
        if KEY_FOR_TAGS in config[key]:
            for tag in config[key][KEY_FOR_TAGS]:
                tags.add(tag)
    return sorted(tags)


def get_tag_names_from_storage(storage_dir):
    config = get_storage_config(storage_dir)
    return get_tag_names(config)


def get_select_tree(storage_dir):
    config = get_storage_config(storage_dir)
    tags = get_tag_names(config)
    all_keys = set()
    nodes = {}
    for tag in tags:
        nodes[tag] = {"label": tag, "value": tag, "children": []}
    for key in config:
        if KEY_FOR_TAGS in config[key]:
            for tag in config[key][KEY_FOR_TAGS]:
                if tag not in nodes:
                    continue
                name = config[key][KEY_FOR_NAME]
                found = False
                for child in nodes[tag]["children"]:
                    if child["label"] == name:
                        found = True
                        break
                if not found:
                    value = get_tagged_name(tag, name)
                    all_keys.add(value)
                    nodes[tag]["children"].append({"label": name, "value": value})
    nodes_out = []
    for node in nodes:
        nodes_out.append(nodes[node])
    return (nodes_out, tags, sorted(all_keys))


def load_index(persist_path, service_context):
    storage_context = StorageContext.from_defaults(persist_dir=persist_path)
    return load_index_from_storage(storage_context, service_context=service_context)


@st.cache_resource
## ServiceContext is unhashable
def get_multi_vector_index(
    selected_names, storage_dir, _service_context, embed_model, names_tagged=False
):
    service_context = _service_context
    config = get_storage_config(storage_dir)
    if selected_names == "all":
        selected_indexes_ids = config.keys()
    else:
        selected_indexes_ids = get_storage_keys_from_names(
            config, selected_names, names_tagged
        )
    all_indexes = []
    for index_id in selected_indexes_ids:
        if config[index_id][ATTRIBUTE_FOR_EMBED_MODEL] != embed_model:
            continue
        if index_id not in config:
            # TODO: Error Out
            continue
        persist_path = os.path.join(storage_dir, config[index_id][ATTRIBUTE_FOR_PATH])
        # TODO: Silently Ignore invalid indexes
        try:
            index = load_index(persist_path, service_context)
            all_indexes.append(index)
        except Exception as e:
            import traceback

            print(e)
            traceback.print_exc()
            continue
    index = MultiVectorStore(
        indexes=all_indexes,
        service_context=service_context,
        embed_model_name=embed_model,
    )
    return index


def put_result_table(question, prompt, res, model):
    import time
    import json
    import hashlib
    import requests
    import sqlite3

    try:
        ts = time.time()
        obj = {}
        obj["question"] = question
        obj["response"] = res.response
        obj["prompt"] = str(prompt)
        obj["metadata"] = str(res.source_nodes)
        obj["model"] = model
        obj["timestamp"] = ts
        json_str = json.dumps(obj, ensure_ascii=False)
        sign = hashlib.sha512(json_str.encode("UTF-8"))
        obj["signature"] = sign.hexdigest()
        r = requests.post(RECORD_URL, json=obj, timeout=3.0)

        if settings.DEBUG_MODE:
            print(obj)

        # conn = sqlite3.connect("result.db")
        # sql = "INSERT INTO result(timestamp, question, prompt, response, metadata, model) values (CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)"
        # conn.execute(sql, [question, str(prompt), res.response, str(res.source_nodes), model])
        # conn.commit()
        # conn.close()
    except Exception as e:
        print(e)
        pass
