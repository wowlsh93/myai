import os
# from dotenv import load_dotenv
# load_dotenv()

#AI
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI


#UI
import streamlit as st


if __name__ == '__main__':

    openai_key = os.getenv("OPENAI_API_KEY")

    chat_model = ChatOpenAI()

    st.title("AI Poet")

    content = st.text_input("Please suggest subject of Poem What you want", "")


    if st.button("Answer"):
        with st.spinner("wait for it .."):
            result2 = chat_model.predict("please poet to me about " + content)
            st.write(result2)





