import os
import time
import argparse
import csv

from index_module import MULTI_STORAGE_DIR
from engine import get_llm_model_config
from engine import get_embed_model_keys
from engine import AuthRecord
from engine import get_query_engine

llm_configs = get_llm_model_config()

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true")
parser.add_argument("--llm", choices=llm_configs.keys(), default="Gemini Pro")
parser.add_argument("--embed", choices=get_embed_model_keys(), default="オンプレ")
parser.add_argument("--storage", default=MULTI_STORAGE_DIR)
parser.add_argument("--openai_key", help="OpenAI API Key")
parser.add_argument("--google_key", help="Google Gemini API Key")
parser.add_argument("--csv", action="store_true")
parser.add_argument("--csv_file", default="output.csv")

args = parser.parse_args()

llm_model_key = args.llm
embed_model_key = args.embed
api_key = args.openai_key
g_api_key = args.google_key
i_auth_endpoint = None
i_project_id = None
i_api_key = None

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
if g_api_key:
    os.environ["GOOGLE_API_KEY"] = g_api_key

auth_obj = AuthRecord(api_key, g_api_key, i_api_key, i_project_id, i_auth_endpoint)

query_engine, llama_debug = get_query_engine(
    auth_obj,
    llm_model_key,
    embed_model_key,
    MULTI_STORAGE_DIR,
    "all",
    True,
)

if args.csv:
    csv_f = open(args.csv_file, "a", encoding="utf-8")
    writer = csv.writer(csv_f)

while True:
    try:
        print("質問：")

        question = input()
        if question == "":
            break

        start_t = time.perf_counter()
        res = query_engine.query(question)
        end_t = time.perf_counter()
        dur = end_t - start_t

        print("回答：")

        if args.csv:
            writer.writerow([question, res.response, dur])
            print("%s,%s,%.2f" % (question, res.response, dur))
        else:
            print("%.2f" % dur)
            print(res.response)
    except EOFError:
        break
    except Exception as e:
        print(e)

csv_f.close()
