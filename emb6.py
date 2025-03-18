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
from playwright.sync_api import sync_playwright
# יצירת אובייקט Document מהתוכן
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter




from dotenv import load_dotenv, find_dotenv


def fetch_full_html(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # הפעלת דפדפן Chromium במצב ראש ללא ממשק

        page = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })

        page.goto(url, timeout=90000)  # פתיחת הדף עם טיימאאוט של 60 שניות

        html_content = page.content()  # שליפת כל ה-HTML
        browser.close()
    return html_content

# כתובת הדף לבדיקה
url = "https://www.digitalforms.co.il/Show_form/7"  # שים כאן את כתובת הדף הרצוי

data = fetch_full_html(url)

os.environ["OPENAI_API_KEY"] = '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'


# 1️⃣ קריאה מדף אינטרנטי

loader = PlaywrightURLLoader(urls=[url])
documents = loader.load()


# 2️⃣ פיצול טקסט כדי שיתאים ל-Embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# יצירת אובייקט Document מהתוכן
document = Document(page_content=data)


# Step 3: Initialize the OpenAI embeddings model
embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(texts, embeddings)


retriever = vectorstore.as_retriever()

# 🔹 שאילתא ראשונה – ל-Embeddings (חיפוש וקטורי בלבד)
query_for_embeddings = "מצא שדות קלט להזנת פרטים אישיים "
query_for_llm = "על בסיס המידע שנמצא, מהם הפרטים שהמשתמש צריך למלא?"
relevant_docs = retriever.get_relevant_documents(query_for_embeddings)

# Step 8: Initialize the ChatOpenAI model
llm = ChatOpenAI(model_name="gpt-4o")  # Or use "gpt-4"

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
# Step 9: Perform a search
response = qa_chain.run(query_for_llm)


query = " אילו צ'קבוקסים יש בדף? "
response = qa_chain.run(query)


query = "  בדוק  אילו פרטים המשתמש צריך למלא בדף הזה "
response = qa_chain.run(query)

query = " האם תוכל לבנות תסריט לבדיקת הדף הזה"
response = qa_chain.run(query)

query = " האם תוכל לבנות תוכנית לבודק אילו דברים הוא צריך לבדוק בדף הזה על מנת שהדף יעבוד תקין "
response = qa_chain.run(query)




# Print the response content
print(response.content)