#Phase 3: Setup Frontend

#1. Setup UI with streamlit(model provider, model, system prompt, web search, query)
import streamlit as st
import requests

st.set_page_config(page_title = "LangGraph Agentic AI", layout="centered")
st.title("AI Chatbot Agent")

system_prompt = st.text_area("Define your Agent", height = 70, placeholder="Enter your system prompt here")

MODEL_NAMES_GROQ = ["llama-3.1-8b-instant","gemma2-9b-it","llama-3.3-70b-versatile"]

provider = "Groq"
selected_model = st.selectbox("Select Groq Model",MODEL_NAMES_GROQ)

allow_web_search = st.checkbox("Allow Web Search")

user_query = st.text_area("Enter your query", height=150, placeholder="Ask anything..")

#2. Connect with Backend via URL
API_URL = "http://127.0.0.1:1200/chat"

if st.button("Ask Agent"):

    if user_query.strip():
        payload = {"model_name":selected_model,
                   "model_provider":provider,
                   "system_prompt":system_prompt,
                   "messages":[user_query],
                   "allow_search":allow_web_search
                   }
        response = requests.post(API_URL,json = payload)
        if response.status_code == 200:
            response_data = response.json()
            if "error" in response_data:
                st.error(response_data['error'])
            else:
                st.subheader("Agent Response")
                st.markdown(response_data)