import os
import faiss
import readline  # for better CLI experience
from typing import List
from langchain import FAISS, InMemoryDocstore
from langchain.agents import Tool, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms.base import BaseLLM

from .auto import AutoAgent


def run_(message, history):
    objective = message

    llm: BaseLLM = ChatOpenAI(
        temperature=0, request_timeout=120)  # type: ignore
    embeddings = OpenAIEmbeddings()  # type: ignore

    """
   Customize the tools the agent uses here. Here are some others you can add:

   os.environ["WOLFRAM_ALPHA_APPID"] = "<APPID>"
   os.environ["SERPER_API_KEY"] = "<KEY>"

   tool_names = ["terminal", "requests", "python_repl", "human", "google-serper", "wolfram-alpha"]
   """

    tool_names = ["python_repl", "terminal"]

    try:
        os.environ["WOLFRAM_ALPHA_APPID"]
        tool_names.append("wolfram-alpha")
    except KeyError:
        pass

    try:
        os.environ["SERPER_API_KEY"]
        tool_names.append("google-serper")
    except KeyError:
        pass

    tools: List[Tool] = load_tools(tool_names, llm=llm)  # type: ignore

    index = faiss.IndexFlatL2(1536)
    docstore = InMemoryDocstore({})
    vectorstore = FAISS(embeddings.embed_query, index, docstore, {})

    agent = AutoAgent.from_llm_and_objectives(
        llm, objective, tools, vectorstore, verbose=False)

    return agent.run()
