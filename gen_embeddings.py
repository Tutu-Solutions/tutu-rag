#
# Created on Wed Jun 28 2023
#
# Copyright (c) 2023 TuTu Solutions LLC
#

from pathlib import Path
from llama_index.constants import DEFAULT_CHUNK_SIZE
from llama_index.indices.vector_store import GPTVectorStoreIndex
from llama_index import download_loader, LLMPredictor, ServiceContext
from llama_index import StorageContext, load_index_from_storage
from llama_index import SimpleDirectoryReader
from langchain.chat_models import ChatOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
    CBEventType,
)

import os
import codecs
import openai
import sys
import argparse
import mimetypes

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document

from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

## Cached Embedding

OPENAI_EMBED_MODEL="text-embedding-3-large"
GEMINI_EMBED_MODEL="models/embedding-001"
#LOCAL_EMBED_MODEL="intfloat/multilingual-e5-large"
LOCAL_EMBED_MODEL="intfloat/multilingual-e5-small"

class PandasExcelReader(BaseReader):
    r"""Pandas-based CSV parser.

    Parses CSVs using the separator detection from Pandas `read_csv`function.
    If special parameters are required, use the `pandas_config` dict.

    Args:

        pandas_config (dict): Options for the `pandas.read_excel` function call.
            Refer to https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html
            for more information. Set to empty dict by default, this means defaults will be used.

    """

    def __init__(
        self,
        *args: Any,
        pandas_config: dict = {},
        concat_rows: bool = True,
        row_joiner: str = "\n",
        **kwargs: Any
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._pandas_config = pandas_config
        self._concat_rows = concat_rows
        self._row_joiner = row_joiner

    def load_data(
        self,
        file: Path,
        include_sheetname: bool = False,
        sheet_name: Optional[Union[str, int]] = None,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Parse file and extract values from a specific column.

        Args:
            file (Path): The path to the Excel file to read.
            column_name (str): The name of the column to use when creating the Document objects.
        Returns:
            List[Document]: A list of`Document objects containing the values from the specified column in the Excel file.
        """
        import itertools

        import pandas as pd

        import llama_index
        tokenizer = llama_index.get_tokenizer()

        df = pd.read_excel(file, sheet_name=sheet_name, **self._pandas_config)

        keys = df.keys()

        document_list = []

        for key in keys:
            header = ", ".join(df[key].columns.astype(str).tolist())
            cur_text = key +"\n" + header
            for row in df[key].values.astype(str).tolist():
              next_text = cur_text + "\n" + ", ".join(row)
              if len(tokenizer(next_text)) > DEFAULT_CHUNK_SIZE:
                  document_list.append(Document(text=cur_text, extra_info=extra_info or {}))
                  cur_text = key +"\n" + header +"\n" + ", ".join(row)
              else:
                  cur_text = next_text
            document_list.append(Document(text=cur_text, extra_info=extra_info or {}))
        return document_list

def gen_embeddings(file_path, use_local_embed, use_gemini_embed):
    from langchain.storage import LocalFileStore
    from langchain.embeddings import CacheBackedEmbeddings
    from langchain_openai import OpenAIEmbeddings
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    
    embed_store = LocalFileStore("./embed_cache/")

    if use_local_embed:
        model_name = LOCAL_EMBED_MODEL
        underlying_embed_model = HuggingFaceEmbeddings(model_name=model_name)
    elif use_gemini_embed:
        with open("G_API_KEY.txt", "r", encoding="UTF-8") as f:
            os.environ["GOOGLE_API_KEY"] = f.readline().strip()

        model_name = GEMINI_EMBED_MODEL
        underlying_embed_model = GoogleGenerativeAIEmbeddings(model=model_name)
    else:
        with open("API_KEY.txt", "r", encoding="UTF-8") as f:
            openai.api_key = f.readline().strip()

        os.environ["OPENAI_API_KEY"] = openai.api_key
        model_name = OPENAI_EMBED_MODEL
        underlying_embed_model = OpenAIEmbeddings(model=model_name)

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
                underlying_embed_model, embed_store, namespace=model_name
                )

    service_context = ServiceContext.from_defaults(embed_model=LangchainEmbedding(cached_embedder), llm=None, callback_manager=callback_manager)

    try:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context, service_context=service_context)
    except Exception as e:
        index = GPTVectorStoreIndex([], service_context=service_context)

    ftype = mimetypes.guess_type(file_path)[0]

    print(ftype)
    if ftype == "text/csv":
        gen_embeddings_csv(file_path, index, service_context)
    elif ftype == "application/pdf":
        gen_embeddings_pdf(file_path, index, service_context)
    elif (
        ftype
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        gen_embeddings_docx(file_path, index, service_context)
    elif ftype == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        gen_embeddings_xlsx(file_path, index, service_context)
    elif (
        ftype
        == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    ):
        gen_embeddings_pptx(file_path, index, service_context)
    else:
        gen_embeddings_txt(file_path, index, service_context)

    index.storage_context.persist()

    event_pairs = llama_debug.get_event_pairs(CBEventType.EMBEDDING)
    #for pair in event_pairs:
    #    print(pair[1].payload['chunks'])
    #    print("---")
    #    print(pair[1].payload['embeddings'])
    #    print("")


def get_utf8_path(file_path):
    convert_jis_to_utf = False
    while True:
        try:
            path = file_path
            if convert_jis_to_utf:
                path = file_path + "utf-8.csv"
                with codecs.open(file_path, "r", "shift-jis") as src:
                    with codecs.open(path, "w", "utf-8") as dst:
                        while True:
                            data = src.read(1 << 11)
                            if not data:
                                break
                            dst.write(data)
            documents = SimpleDirectoryReader(input_files=[path]).load_data()
        except UnicodeDecodeError:
            print("UTF-8ファイルではありません、shift-jis で再試行します。")
            convert_jis_to_utf = True
        except Exception as e:
            print(e)
            print("UTF-8 / SHIFT-JISファイルではありません。")
            exit(-1)
        else:
            return path


class SimpleCSVReader(BaseReader):
    def __init__(
        self,
        *args: Any,
        concat_rows: bool = True,
        encoding: str = "utf-8",
        **kwargs: Any
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._concat_rows = concat_rows
        self._encoding = encoding

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file."""
        import csv

        text_list = []
        with open(file, "r", encoding=self._encoding) as fp:
            csv_reader = csv.reader(fp)
            for row in csv_reader:
                text_list.append(", ".join(row))
        if self._concat_rows:
            return [Document(text="\n".join(text_list), extra_info=extra_info or {})]
        else:
            return [
                Document(text=text, extra_info=extra_info or {}) for text in text_list
            ]

    def load_node_with_headers(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        import csv

        import llama_index
        tokenizer = llama_index.get_tokenizer()

        document_list = []
        with open(file, "r", encoding=self._encoding) as fp:
            csv_reader = csv.reader(fp)
            header = ", ".join(next(csv_reader))
            cur_text = header
            for row in csv_reader:
                next_text = cur_text + "\n" + ", ".join(row)
                if len(tokenizer(next_text)) > DEFAULT_CHUNK_SIZE:
                    print(cur_text)
                    document_list.append(Document(text=cur_text, extra_info=extra_info or {}))
                    cur_text = header + "\n" + ", ".join(row)
                else:
                    cur_text = next_text
            print(cur_text)
            document_list.append(Document(text=cur_text, extra_info=extra_info or {}))
        return document_list


def gen_embeddings_csv(file_path, index, service_context):
    path = get_utf8_path(file_path)

    loader = SimpleCSVReader()

    documents = loader.load_node_with_headers(file=Path(path))

    for document in documents:
        index.update(document, service_context=service_context)


def gen_embeddings_pdf(file_path, index, service_context):
    CJKPDFReader = download_loader("CJKPDFReader")

    loader = CJKPDFReader()

    documents = loader.load_data(file=Path(file_path))

    for document in documents:
        index.update(document, service_context=service_context)


def gen_embeddings_docx(file_path, index, service_context):
    DocxReader = download_loader("DocxReader")

    loader = DocxReader()

    documents = loader.load_data(file=Path(file_path))

    for document in documents:
        index.update(document, service_context=service_context)


def gen_embeddings_xlsx(file_path, index, service_context):
    # PandasExcelReader = download_loader("PandasExcelReader")

    loader = PandasExcelReader()

    documents = loader.load_data(file=Path(file_path), include_sheetname=True)

    for document in documents:
        index.update(document, service_context=service_context)


def gen_embeddings_pptx(file_path, index, service_context):
    PptxReader = download_loader("PptxReader")

    loader = PptxReader()

    documents = loader.load_data(file=Path(file_path))

    for document in documents:
        index.update(document, service_context=service_context)


def gen_embeddings_txt(file_path, index, service_context):
    path = get_utf8_path(file_path)

    documents = SimpleDirectoryReader(input_files=[path]).load_data()

    for document in documents:
        index.update(document, service_context=service_context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-l", "--local_embed", action="store_true")
    parser.add_argument("-g", "--gemini_embed", action="store_true")

    args = parser.parse_args()

    if args.debug:
        import logging
        import sys

        logging.basicConfig(level=logging.DEBUG, filename="gen_embeddings.log")
    gen_embeddings(args.filename, args.local_embed, args.gemini_embed)
