import os
from dotenv import load_dotenv


#AI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA

#UI
import streamlit as st


if __name__ == '__main__':
    load_dotenv()
    loader = PyPDFLoader("unsu.pdf")
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.split_documents(pages)

    embeddings_model = OpenAIEmbeddings()

    db = Chroma.from_documents(texts, embeddings_model)


    #question!!
    question = "내용을 요약해줘"
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
    result = qa_chain({"query": question})
    print(result)







