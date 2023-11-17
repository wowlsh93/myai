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

load_dotenv()

st.header('PDF Genie')
st.write('질문하실 PDF를 업로드해주세요.')
st.write('---')


# # Make a temp folder can store uploaded file.

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

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


with st.sidebar:
    st.header('파일업로드')
    uploaded_files = st.file_uploader(
        '파일을 선택해주세요.',
        accept_multiple_files=True
    )

    # uploaded_file_no = (uploaded_file)
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

            st.write(bytes_data.name, 'File DONE')

# show input box
question = st.text_input('질문을 입력해주세요.')
if st.button('궁금해', type="primary"):
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
        # st.write('질문 :', question)
        # st.write('답변 :', answer)

print('DONE')





