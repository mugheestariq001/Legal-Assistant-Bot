import streamlit as st
from LLMRAGModel import model
import torch
import gc

def authenticate(username, password):
    # Add your authentication logic here
    return username == "admin" and password == "password"

def query_llm():
    llm_chain, foo = model.getnewChain()
    st.session_state['llm_chain'] = llm_chain
    st.write(f"Current user is: {st.session_state['llm_chain']}")

def main():
    st.title("LLM RAG Model")

    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate(username, password):
                st.session_state['authenticated'] = True
                st.success("Logged in successfully")
                query_llm()
            else:
                st.error("Invalid username or password")
        return

    st.write("Welcome to the Legal Assistant Bot")

    if 'llm_chain' not in st.session_state:
        query_llm()

    user_input = st.text_input("Enter your query:")
    if st.button("Submit"):
        llm_chain = st.session_state['llm_chain']
        response = llm_chain.invoke(user_input)
        st.write(response['text'])
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
