import streamlit as st
import os
import pdfplumber
import time
import uuid

import gspread
from google.oauth2.service_account import Credentials

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_tavily import TavilySearch
from datetime import datetime
import pytz

# -----------------------------
# Initialize API keys from secrets
# -----------------------------
try:
    groq_api_key = st.secrets["GROQ"]["API_KEY"]
    tavily_api_key = st.secrets["TAVILY"]["API_KEY"]
    os.environ["TAVILY_API_KEY"] = tavily_api_key
except KeyError:
    st.error("API keys not found in .streamlit/secrets.toml. Please add them correctly.")
    st.stop()

# -----------------------------
# Google Sheets Client Setup
# -----------------------------
def get_gsheet_client():
    """Authenticates with Google Sheets and returns the client and sheet ID."""
    try:
        credentials_dict = st.secrets["gcp_service_account"]
        sheet_id = credentials_dict["sheet_id"]
        
        credentials = Credentials.from_service_account_info(
            credentials_dict,
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        client = gspread.authorize(credentials)
        return client, sheet_id
    except Exception as e:
        st.error(f"Failed to authenticate with Google Sheets: {e}")
        return None, None

def log_performance(query_id, mode, query, response_text, duration, status, error_code="", error_message=""):
    """Logs the performance metrics to a Google Sheet."""
    client, sheet_id = get_gsheet_client()
    if not client:
        return

    try:
        sheet = client.open_by_key(sheet_id).sheet1
        india_tz = pytz.timezone("Asia/Kolkata")
        timestamp = datetime.now(india_tz).strftime('%Y-%m-%d %H:%M:%S')
        row = [
            query_id,
            timestamp,
            mode,
            query,
            len(response_text) if isinstance(response_text, str) else 0,
            int(duration * 1000),
            status,
            error_code,
            error_message
        ]
        sheet.append_row(row)
    except Exception as e:
        st.error(f"Error logging data to Google Sheets: {e}")

# -----------------------------
# Define a single LLM instance
# -----------------------------
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# -----------------------------
# PDF Processing Functions
# -----------------------------
@st.cache_data
def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def trim_text(text, max_tokens=3000):
    """Trims the text to a specified number of tokens."""
    tokens = text.split()
    if len(tokens) > max_tokens:
        return ' '.join(tokens[:max_tokens])
    return text

@st.cache_resource
def process_pdf_with_langchain(pdf_text):
    """Processes PDF text to create a LangChain QA chain."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    documents = text_splitter.create_documents([pdf_text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain

# -----------------------------
# Research & Math Agents
# -----------------------------
def handle_research(query):
    """Handles general research queries using a search agent."""
    search = TavilySearch(max_results=3)
    tools = [
        Tool(
            name="Tavily Search",
            description="A search tool to get the latest information from the web.",
            func=search.invoke
        )
    ]
    research_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    try:
        response = research_agent.run(query)
        return response, "Success", "", ""
    except Exception as e:
        return f"An error occurred while running the search agent: {e}", "Fail", "429", str(e)

def handle_math(query):
    """Solves a math problem step-by-step."""
    try:
        prompt_template = PromptTemplate.from_template(
            "You are a math tutor. Solve the following problem step-by-step and provide the final answer. Explain your reasoning clearly: {query}"
        )
        prompt = prompt_template.format(query=query)
        response = llm.invoke(prompt)
        return getattr(response, "content", response), "Success", "", ""
    except Exception as e:
        return f"An error occurred while solving the math problem: {e}", "Fail", "500", str(e)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="BrainyBot", page_icon="🤖")

st.title("BrainyBot: Your Intelligent Assistant")
st.markdown("This app can answer questions about a PDF, perform general research, and solve math problems.")

mode = st.selectbox(
    "Choose a mode:",
    ("PDF Q&A", "Research & Latest Topics", "Solve Math Problem")
)

user_query = st.text_input("Enter your query:", key="user_query_input")

if user_query:
    query_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    with st.spinner("Processing your request..."):
        response = ""
        status = "Success"
        error_code = ""
        error_message = ""

        try:
            if mode == "PDF Q&A":
                pdf_file = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_uploader")
                if pdf_file:
                    pdf_text = extract_text_from_pdf(pdf_file)
                    trimmed_pdf_text = trim_text(pdf_text)

                    if trimmed_pdf_text:
                        qa_chain = process_pdf_with_langchain(trimmed_pdf_text)
                        response = qa_chain({"query": user_query})['result']
                    else:
                        response = "Could not extract text from the PDF."
                        status = "Fail"
                        error_code = "400"
                        error_message = "Empty PDF text"
                else:
                    response = "No PDF uploaded."
                    status = "Fail"
                    error_code = "400"
                    error_message = "Missing PDF file"

            elif mode == "Research & Latest Topics":
                response, status, error_code, error_message = handle_research(user_query)

            elif mode == "Solve Math Problem":
                response, status, error_code, error_message = handle_math(user_query)

        except Exception as e:
            response = f"Unexpected error: {e}"
            status = "Fail"
            error_code = "500"
            error_message = str(e)

    duration = time.perf_counter() - start_time

    st.markdown("---")
    if response:
        st.write("### Response:")
        st.write(response)

    log_performance(query_id, mode, user_query, response, duration, status, error_code, error_message)

st.markdown("---")
if st.button("Clear App"):
    st.session_state.clear()
    st.experimental_rerun()
