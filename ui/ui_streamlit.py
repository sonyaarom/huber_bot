import sys
import os
cwd = os.getcwd()

sys.path.insert(0, os.path.abspath(os.path.join(cwd, '../scripts')))
sys.path.insert(0, os.path.abspath(os.path.join(cwd, '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(cwd, '../models')))
sys.path.insert(0, os.path.abspath(os.path.join(cwd, '../img')))
import streamlit as st

from model_management.model import Model
from pathlib import Path
import time
import numpy as np
import dotenv


MODEL_WEIGHTS_FILENAME = "mistral-7b-instruct-v0.1.Q3_K_M.gguf"
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
INDEX_NAME = "data-management-example"
#LOGO_IMAGE = '/Users/s.konchakova/Thesis/img/hu-berlin-logo.png'

def initialise_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model" not in st.session_state:
        weights_file = "../models/" + MODEL_WEIGHTS_FILENAME
        st.session_state.model = Model(model_path=weights_file, index_name=INDEX_NAME, pinecone_api_key=PINECONE_API_KEY)

def app():
    #st.sidebar.image(LOGO_IMAGE, width=150, use_column_width=False)
    st.sidebar.title('How to Use')
    st.sidebar.markdown("""
        1. Write your question in the input box.
        2. Ensure your question is clear and concise.
        3. Provide context and specific details when asking.
        4. Please note that the chatbot is still learning! As it might hallucinate, please double-check its answers.
        5. If there are any questions, complaints, or suggestions, refer here: *s.konchakova@student.hu-berlin.de*
    """)

    st.sidebar.error('Bad example: What is this?')
    st.sidebar.success('Good example: When is the semester break for winter semester 2024?')

    st.title('HUBer - AI assistant of HU Berlin!')
    
    # Clean History button
    if st.sidebar.button(label='Clear History'):
        st.session_state.messages = []  # Clear chat history

    initialise_session_state()

    predefined_responses = {
        "What is HUBer?": 
            "HUBer is a work of student, chatbot that can answer WiWi-HU's related questions.",
        
        "How to use HUBer?": 
            "Ganz easy!",

        "What languages does HUBer speak?": 
            "For now HUBer can answer questions in English. "
    }

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask us anything :)"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if prompt in predefined_responses:
            time.sleep(np.random.randint(5,15))
            response = predefined_responses[prompt]
        else:
            response = st.session_state.model.process_query(query=prompt, num_chunks=2, num_urls=2)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    app()