import tempfile
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


st.set_page_config(page_title="Underwriting assistance", layout="wide")

st.title("📄 Underwriting assistance")

question = st.text_input(
    "Ask a question about the Underwriting"
)

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

print(f"OpenAI API Key: {openai_api_key}")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")


if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(splits, embeddings, persist_directory="./chroma_db")

    st.write(f"Loaded {len(docs)} pages")
    st.write(f"Created {len(splits)} chunks")

    # print(f"Loaded {len(docs)} documents.")

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # splits = text_splitter.split_documents(docs)

    # print(f"Created {len(splits)} text chunks.")

    # Create embeddings and store in a local Chroma database


    # 1. Setup Prompt & Model
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "You are a underwriter. you should verify the documents and you should answer the valid or invalid documents.kwargs={context}\n\nQuestion: {question}"
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
