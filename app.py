import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS # vector database 
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings # vector embedding model
import requests
import asyncio
import nest_asyncio

from dotenv import load_dotenv


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


st.title("Gemma Document Q&A")

llm = ChatGroq(api_key=groq_api_key, 
               model="gemma2-9b-it")


prompt = ChatPromptTemplate.from_template(
    """ 
    answer the question based on the provided context.
    please provide the best accurate response based on the question.
    <context>
    {context}
    <context>
    
    Question: {input}  
    
    """           
)    


def vector_embeddings():
    
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    nest_asyncio.apply()
    
    if "vectors" not in st.session_state:
        
        """  
        1.load an embedding model for create embeddings. Loads google embedding-001 model
        2.load the documents from a directory. Loader catch the directory by PyPDFDirectoryLoader and then load all of them from directory.
        3. Create a splitter where describe chunk_size and chunk_overlap.
        4. Split loaded documents into smaller chunks.
        5. Create a vector database by FAISS for my embeddings. Name: vectorstore
        6. Now I can use this 'vectorstore' for retrieval based on the context of the
        """
        
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("D:/krish-rag-app/us_census") # data ingestion
        st.session_state.docs = st.session_state.loader.load() # data loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # text splitting
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs) # text splitting
        st.session_state.vectorstore = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings) # vector database
        

prompt1 = st.text_input("Ask a question about the US Census data")

if st.button("Create Vector DB"):
    vector_embeddings()
    st.write("DB is prepared")        
    

import time

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt) 
    retriever = st.session_state.vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt1})
    st.write(response["answer"])
    
    # streamlit expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------------------------")   