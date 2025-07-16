import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import datetime
from app.qa_chain import get_qa_chain

# Set page config and background
st.set_page_config(page_title="Microsoft Admin Center AI", layout="wide")

# Inject CSS for black background and white text on the whole page
st.markdown(
    """
    <style>
    .main {
        background-color: #000000;
        color: white;
        min-height: 100vh;
        padding: 1rem 2rem;
    }
    .stTextInput>div>div>input {
        background-color: #222222;
        color: white;
        border: 1px solid #444444;
    }
    .stButton>button {
        background-color: #444444;
        color: white;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛡️ Microsoft Admin Center Assistant")
st.write("Ask questions **only related** to Microsoft Admin Center. Other questions will be politely declined.")

# Initialize the chat chain (load vectorstore + llm)
qa_chain = get_qa_chain()

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Container for chat messages
chat_container = st.container()

# User input form
with st.form(key="input_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question:")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    # Append user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Get AI response
    with st.spinner("Thinking..."):
        result = qa_chain(user_input)
        answer = result["result"]

    # Append AI response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# Display chat history with styling for user and assistant
with chat_container:
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            st.markdown(
                f'<div style="text-align:right; background:#0d0d0d; color:white; padding:10px; border-radius:10px; margin:5px 0; max-width:70%; margin-left:auto;">{content}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="text-align:left; background:#222222; color:white; padding:10px; border-radius:10px; margin:5px 0; max-width:70%;">{content}</div>',
                unsafe_allow_html=True,
            )
