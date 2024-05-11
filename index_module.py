import os
import json

## LlamaIndex Import
from llama_index import StorageContext, load_index_from_storage

## Local Import
from multi_vector_index import MultiVectorStore

## Keyword Definitions
MULTI_STORAGE_DIR = "./multi_storage"
CONFIG_FILE = "config.json"

ATTRIBUTE_FOR_PATH = "path"
ATTRIBUTE_FOR_EMBED_MODEL = "embed_model"

PREFIX_FOR_INDEXES = "index_"
KEY_FOR_INDEXES = "indexes"

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

def get_storage_keys_from_names(config, names):
    names_set = set(names)
    selected_indexes_ids = []
    for storage_key in config.keys():
        if config[storage_key]["file_name"] in names_set:
            selected_indexes_ids.append(storage_key)
    return selected_indexes_ids

def get_multi_vector_index(selected_names, storage_dir, service_context, embed_model):
    config = get_storage_config(storage_dir)
    if selected_names == "all":
        selected_indexes_ids = config.keys()
    else:
        selected_indexes_ids = get_storage_keys_from_names(config, selected_names)
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
            storage_context = StorageContext.from_defaults(persist_dir=persist_path)
            index = load_index_from_storage(storage_context, service_context=service_context)
            all_indexes.append(index)
        except Exception as e:
            print(e)
            continue
    index = MultiVectorStore(indexes=all_indexes, service_context=service_context, embed_model_name=embed_model)
    return index

#RECORD_URL="http://127.0.0.1:5001/gpt-dev-34c73/us-central1/add"
RECORD_URL="https://add-dy5bhzh3sq-uc.a.run.app/add"

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
        sign = hashlib.sha512(json_str.encode('UTF-8'))
        obj["signature"] = sign.hexdigest()
        r = requests.post(RECORD_URL, json=obj, timeout=1.0)

        #conn = sqlite3.connect("result.db")
        #sql = "INSERT INTO result(timestamp, question, prompt, response, metadata, model) values (CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)"
        #conn.execute(sql, [question, str(prompt), res.response, str(res.source_nodes), model])
        #conn.commit()
        #conn.close()
    except Exception:
        pass
