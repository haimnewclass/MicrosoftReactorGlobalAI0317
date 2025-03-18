import os

import numpy as np
from langchain_core.runnables import RunnableLambda
from openai import OpenAI, _exceptions
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


from dotenv import load_dotenv, find_dotenv

os.environ["OPENAI_API_KEY"] = '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'

# Create the ChatOpenAI model
llm1 = ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=os.environ["OPENAI_API_KEY"])
llm2 = ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=os.environ["OPENAI_API_KEY"])

template1 = PromptTemplate.from_template("מהם חמשת הדברים החשובים ביותר לדעת על {subject}?")
template2 = PromptTemplate.from_template("תרגם לסינית: {text}")


# Create the LLM chains
chain1 = template1 | llm1
chain2 = template2 | llm2

convert_output = RunnableLambda(lambda x: {"text": x.content})


sequence = chain1 | convert_output | chain2


response = sequence.invoke({"subject": "תורת היחסות"})

print(response.content)

print("End of lang2.py")