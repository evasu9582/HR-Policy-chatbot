import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import urllib.parse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st


st.set_page_config(page_title="HR Policy Chatbot", layout="wide")

st.title("📄 HR Policy Chatbot")

question = st.text_input(
    "Ask a question about the HR Policy"
)

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

print(f"OpenAI API Key: {openai_api_key}")

loader = PyPDFLoader("D:\\HR-Chatbot\\Documents\\HR-Policy.pdf")
docs = loader.load()

print(f"Loaded {len(docs)} documents.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print(f"Created {len(splits)} text chunks.")

# Create embeddings and store in a local Chroma database
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(splits, embeddings, persist_directory="./chroma_db")

# 1. Setup Prompt & Model
llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = ChatPromptTemplate.from_template(
    "Use the context below to answer the question:\n\nContext: {context}\n\nQuestion: {question}"
)

# 2. Build the retriever helper
retriever = retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10
    }
)

def format_docs(docs):
    return "\n\n".join(doc.page_content[:1000] for doc in docs)

# 3. Create the pure LCEL pipeline
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. Invoke seamlessly
if question:
    with st.spinner("Processing..."):
        answer = rag_chain.invoke(question)
        st.success("Done!")
        st.write(answer)
