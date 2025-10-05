# BrainyBot: Multi-Agent Chatbot for PDFs, Math, and Research

## Overview

**BrainyBot** is a multi-agent AI chatbot built using the **Groq API** and **LangChain framework**. It can:

* Answer questions from uploaded PDFs (**PDF Q&A**)
* Solve mathematical and logical problems (**Math Agent**)
* Retrieve and summarize the latest information from the web (**Research Agent**)

The system tracks performance metrics for every request and logs them to **Google Sheets** for analysis.

---

## Features

* **PDF Q&A**: Upload a PDF and ask questions. Utilizes FAISS vector database and HuggingFace embeddings.
* **Math Problem Solver**: Solves arithmetic, algebra, probability, and calculus problems step-by-step.
* **Research Agent**: Fetches the latest online information using TavilySearch API.
* **Performance Logging**: Logs query, response length, execution time, and status to Google Sheets.
* **Streamlit Interface**: Easy-to-use web interface for interaction.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/himanshuPabbi/Chatbot-Fast-Latest.git
cd Chatbot-Fast-Latest
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure API keys in `.streamlit/secrets.toml`:

```toml
[GROQ]
API_KEY = "your_groq_api_key"

[TAVILY]
API_KEY = "your_tavily_api_key"

[gcp_service_account]
type = "service_account"
project_id = "your_project_id"
private_key_id = "your_private_key_id"
private_key = "your_private_key"
client_email = "your_client_email"
client_id = "your_client_id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "your_cert_url"
sheet_id = "your_google_sheet_id"
```

---

## Usage

1. Run the Streamlit app:

```bash
streamlit run src/app.py
```

2. Select a **mode**:

   * PDF Q&A
   * Research & Latest Topics
   * Solve Math Problem

3. Enter your query and optionally upload a PDF (for PDF Q&A).

4. View the response on the interface.

5. All requests are logged automatically in Google Sheets.

---

## Reproducing Results

* The system logs performance metrics for each query:

  * Query ID
  * Mode
  * Response length
  * Execution time (ms)
  * Status (Success/Fail)
  * Error codes
* These logs can be analyzed to replicate Tables and Figures from the research paper.

---

## Notes

* Ensure API rate limits are considered when running multiple requests in **Research & Latest Topics** mode.
* For large PDFs, the text is automatically trimmed to 3000 tokens.
* The app is optimized for **Streamlit Cloud**, but can be run locally.

---

---

