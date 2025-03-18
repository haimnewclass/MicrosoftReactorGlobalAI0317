import os

# pip install faiss-cpu
import numpy as np
from openai import OpenAI, _exceptions
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


from dotenv import load_dotenv, find_dotenv


os.environ["OPENAI_API_KEY"] = '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'

# Create the ChatOpenAI model
llm = ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=os.environ["OPENAI_API_KEY"])


response = llm.invoke("מהי תורת היחסות?")
print(response)

print("End of lang1.py")