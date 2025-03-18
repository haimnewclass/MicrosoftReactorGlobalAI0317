from sqlalchemy import create_engine
from langchain.utilities import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_sql_agent, AgentType
import os
import langchain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from sqlalchemy import create_engine,text
import urllib
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents import AgentType

import numpy as np
print("starting")


langchain.debug = True
# יצירת מחרוזת חיבור נכונה
connection_string = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=NC1\\SQLEXPRESS;"
    "DATABASE=Northwind;"
    "Trusted_Connection=yes;"
)

# קידוד המחרוזת כך שתעבוד עם SQLAlchemy
encoded_connection_string = urllib.parse.quote_plus(connection_string)

# יצירת URI תקין
db_uri = f"mssql+pyodbc:///?odbc_connect={encoded_connection_string}"

# יצירת מנוע SQLAlchemy
engine = create_engine(db_uri)

# בדיקת חיבור
with engine.connect() as conn:
    result = conn.execute(text("SELECT 1"))
    print(result.fetchone())  # אמור להחזיר (1,)

os.environ["OPENAI_API_KEY"] = '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'


# יצירת אובייקט SQLDatabase
database = SQLDatabase(engine)

# הגדרת מודל LLM (OpenAI או מודל מקומי)
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=os.environ["OPENAI_API_KEY"])

toolkit = SQLDatabaseToolkit(db=database, llm=llm)
agent = create_sql_agent(llm=llm, toolkit=toolkit, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbos=True
,enable_cache = True  # הפעלת Cache פנימי
)

# ביצוע שאילתת SQL דינמית בשפה טבעית
query = " מהו המוצר הכי מבוקש  במדינות שהן אינן אירופה   "
response = agent.invoke(query)
print(response)
