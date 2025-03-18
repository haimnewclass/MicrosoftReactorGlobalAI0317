import os
import faiss
# pip install faiss-cpu
import numpy as np
from openai import _exceptions
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PlaywrightURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from dotenv import load_dotenv, find_dotenv


os.environ["OPENAI_API_KEY"] = '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'


# 1️⃣ קריאה מדף אינטרנטי
#url = "https://b2b.btl.gov.il/BTL.ILG.Payments/BankMaanakForm.aspx"  # שים כאן את כתובת הדף הרצוי
url  = "https://edition.cnn.com/2025/03/17/politics/trump-putin-meeting-ukraine-intl-hnk/index.html"
loader = PlaywrightURLLoader(urls=[url])
documents = loader.load()


# 2️⃣ פיצול טקסט כדי שיתאים ל-Embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)


# Step 3: Initialize the OpenAI embeddings model
embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(texts, embeddings)


retriever = vectorstore.as_retriever()

# Step 8: Initialize the ChatOpenAI model
llm = ChatOpenAI(model_name="gpt-4o")  # Or use "gpt-4"

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
# Step 9: Perform a search
query = " סכם את הכתבה המרכזית ואת הכתבה המישנית "

response = qa_chain.run(query)





# Print the response content
print(response.content)