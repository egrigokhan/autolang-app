import os

from langchain import ConversationChain, LLMChain, OpenAI, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import (AIMessagePromptTemplate,
                                    ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from lib.index import run_

import json


def approve(type, request):
    #  switch case
    return "Approved."


def run(message, history):
    try:
        return run_(message, history)
    except Exception as e:
        print(e)
        return "I'm sorry, I'm having trouble understanding you. Could you please rephrase?"


def setup(config):
    os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
    try:
        os.environ["WOLFRAM_ALPHA_APPID"] = config["WOLFRAM_ALPHA_APPID"]
    except:
        pass
    try:
        os.environ["SERPER_API_KEY"] = config["SERPER_API_KEY"]
    except:
        pass
