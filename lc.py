import settings

from keys import *
from index_module import MULTI_STORAGE_DIR
from engine import get_index
from engine import get_embed_model
from engine import get_embed_model_by_key

## LlamaIndex Import
from llama_index.legacy import ServiceContext, StorageContext
from llama_index.legacy import SimpleDirectoryReader

## Langchain Import
from langchain_community.retrievers.llama_index import LlamaIndexRetriever

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true")
parser.add_argument("-k", "--top_k", type=int, default=settings.TOP_SIMIRALITY_K)
parser.add_argument("-p", "--prompt", default=settings.PROMPT_ID)

args = parser.parse_args()

settings.DEBUG_MODE = args.debug
settings.TOP_SIMIRALITY_K = args.top_k
settings.PROMPT_ID = args.prompt

import os

import langchain
from langchain.cache import SQLiteCache

langchain.llm_cache = SQLiteCache(database_path=".lc.db")

embed_model_key = EMBED_KEY_ONPREMISE

index, _, _ = get_index(embed_model_key, MULTI_STORAGE_DIR, "all", True)

embed_model = get_embed_model_by_key(embed_model_key)
(embed_model, embed_model_name) = get_embed_model(embed_model)
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)
retriever = LlamaIndexRetriever(
    index=index.as_query_engine(
        service_context=service_context, similarity_top_k=settings.TOP_SIMIRALITY_K
    )
)

from typing import Dict, TypedDict
from langchain_core.messages import BaseMessage


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]


import json
import operator
from typing import Annotated, Sequence, TypedDict

import bigframes.dataframe

from langchain import hub
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import PydanticFunctionsOutputParser

### Nodes ###


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = retriever.get_relevant_documents(question)
    return {"keys": {"documents": documents, "question": question}}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Prompt
    # prompt = hub.pull("rlm/rag-prompt")

    prompt = PromptTemplate(
        template="""以下の情報を参照してください。\n
        ---------------------\n
        {context}\n
        ---------------------\n
        この情報を使って、次の質問に日本語で答えてください。: {question} \n""",
        input_variables=["context", "question"],
    )

    # LLM
    llm = ChatGoogleGenerativeAI(model="gemini-pro", streaming=True, temperature=0)
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {
        "keys": {"documents": documents, "question": question, "generation": generation}
    }


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with relevant documents
    """

    print("---CHECK RELEVANCE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)

    # Tool
    grade_tool_oai = convert_to_openai_tool(grade)

    # LLM with tool and enforce invocation
    llm_with_tool = model.bind(
        tools=[grade_tool_oai],
        tool_choice={"type": "function", "function": {"name": "grade"}},
    )

    # Parser
    parser_tool = PydanticToolsParser(tools=[grade])

    # LLM
    # modelg = ChatGoogleGenerativeAI(model="gemini-pro", streaming=True, tools=[grade], temperature=0)
    modelg = ChatGoogleGenerativeAI(model="gemini-pro", streaming=True, temperature=0)

    # Tool
    grade_tool_oaig = convert_to_openai_tool(grade)

    # LLM with tool and enforce invocation
    # llm_with_tool = model.bind_tools([grade])
    llm_with_toolg = modelg.bind(tools=[grade])
    # llm_with_toolg = modelg.bind(tools=[grade_tool_oaig])

    # Parser
    parser_toolg = PydanticFunctionsOutputParser(pydantic_schema={"grade": grade})

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Reply 'yes' or 'no' to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    # chain = prompt | llm_with_tool | parser_tool
    # chain = prompt | llm_with_tool | StrOutputParser()

    chaing = prompt | model | StrOutputParser()
    # chain2 = prompt | llm_with_tool

    # Score
    filtered_docs = []
    search = "No"  # Default do not opt for web search to supplement retrieval
    for d in documents:
        # scoreg = chaing.invoke({"question": question, "context": d.page_content})
        grade = chaing.invoke({"question": question, "context": d.page_content})
        print(grade)
        # grade = scoreg.binary_score
        # score = chain.invoke({"question": question, "context": d.page_content})
        # print(score)
        # grade = score[0].binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            search = "Yes"  # Perform web search
            continue

    ## Only perform web search when no relevant doc
    search = "Yes" if len(filtered_docs) == 0 else "No"

    return {
        "keys": {
            "documents": filtered_docs,
            "question": question,
            "run_web_search": search,
        }
    }


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Create a prompt template with format instructions and the query
    prompt = PromptTemplate(
        template="""You are generating questions that is well optimized for retrieval. \n
        Look at the input and try to reason about the underlying sematic intent / meaning. \n
        Here is the initial question:
        \n ------- \n
        {question}
        \n ------- \n
        Formulate an improved question in the same Language: """,
        input_variables=["question"],
    )

    # Grader
    model = ChatGoogleGenerativeAI(
        model="models/gemini-pro", streaming=True, temperature=0
    )
    # model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)

    # Prompt
    chain = prompt | model | StrOutputParser()
    better_question = chain.invoke({"question": question})
    print(better_question)

    return {
        "keys": {
            "documents": documents,
            "question": better_question,
            "ori_question": question,
        }
    }


def web_search(state):
    """
    Web search based on the re-phrased question using Tavily API.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    state_dict = state["keys"]
    question = state_dict["question"]
    ori_question = state_dict["ori_question"]
    documents = state_dict["documents"]

    tool = TavilySearchResults()
    docs = tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"keys": {"documents": documents, "question": ori_question}}


### Edges


def decide_to_generate(state):
    """
    Determines whether to generate an answer or re-generate a question for web search.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Next node to call
    """

    print("---DECIDE TO GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    filtered_documents = state_dict["documents"]
    search = state_dict["run_web_search"]

    if search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TRANSFORM QUERY and RUN WEB SEARCH---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


import pprint

from langgraph.graph import END, StateGraph

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("web_search", web_search)  # web search

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

while True:
    question = input()
    if question == "":
        break
    # Run
    inputs = {"keys": {"question": question}}
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint.pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")

    # Final generation
    pprint.pprint(value["keys"]["generation"])
