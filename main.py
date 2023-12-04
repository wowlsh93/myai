__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Optional, Union
from uuid import UUID
from dotenv import load_dotenv
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import tempfile
import os
# from PIL import Image

from abc import ABC, abstractmethod
from typing import Any, List
from langchain.schema import Document
from langchain.callbacks.manager import Callbacks

from langchain.retrievers.web_research import WebResearchRetriever
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chains import RetrievalQAWithSourcesChain


# load_dotenv()

os.environ["GOOGLE_CSE_ID"] = "f3f9e0f4d4b984777"
os.environ["GOOGLE_API_KEY"] = "AIzaSyDWt5BsWSo0DdZtIBfA5QRVFNzwpTbnKgE"
os.environ["OPENAI_API_KEY"] = "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"



#Make a temp folder can store uploaded file.
def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load file and Split by "page".
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    print('file loaded')
    return pages


class StreamingHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        pass

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)
        pass

if __name__ == "__main__":

    st.header("TT Service")

    pdftab, webtab = st.tabs(["PDF", "WEB"])

    with pdftab:
        pdftab.subheader("Chat with PDF!!")

        uploaded_files = st.file_uploader(
            'PDF를 업로드해주세요.',
            accept_multiple_files=True
        )
        if uploaded_files:
            # load text
            for uploaded_file in uploaded_files:
                bytes_data = uploaded_file
                pages = pdf_to_document(bytes_data)

                # split text
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=20,
                    length_function=len,
                    # add_start_index=True,
                )
                texts = text_splitter.split_documents(pages)

                # embedding and store
                embedding_model = OpenAIEmbeddings()
                db = Chroma.from_documents(texts, embedding_model)

                st.write(bytes_data.name, ' is uploaded!!')

        st.write('---')
        # show input box

        question = st.text_input('질문을 입력해주세요.',key="1_1")
        if st.button('궁금해', type="primary", key="1_2"):
            with st.spinner('처리중입니다'):
                chat_box = st.empty()
                stream_hander = StreamingHandler(chat_box)
                myllm = ChatOpenAI(
                    temperature=0,
                    max_tokens=100,
                    streaming=True,
                    callbacks=[stream_hander],
                )
                qa = RetrievalQA.from_chain_type(
                    llm=myllm,
                    retriever=db.as_retriever()
                )
                answer = qa.run(question)
                qa.run(question)

        print('DONE')


    with webtab:
        webtab.subheader("Chat with Web-Link")

        # Vectorstore 셋팅하기
        vectorstore = Chroma(embedding_function=OpenAIEmbeddings(),
                             persist_directory="./chroma_db_oai")


        question = st.text_input('질문을 입력해주세요.', key="2_1")
        if st.button('궁금해', type="primary", key="2_2"):
            with st.spinner('처리중입니다'):
                chat_box = st.empty()
                stream_hander = StreamingHandler(chat_box)
                search_llm = ChatOpenAI(
                    temperature=0,
                    max_tokens=100,
                    streaming=True,
                    callbacks=[stream_hander],
                )

                search = GoogleSearchAPIWrapper()

                web_research_retriever = WebResearchRetriever.from_llm(
                    vectorstore=vectorstore,
                    llm=search_llm,
                    search=search,
                )

                response_llm = ChatOpenAI(temperature=0.90)
                qa_chain = RetrievalQAWithSourcesChain.from_chain_type(response_llm,
                                                                       retriever=web_research_retriever)
                result = qa_chain({"question": question})
                st.write(result)

        print('DONE')










