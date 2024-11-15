import os
import time
import tempfile
import streamlit as st
from streamlit_chat import message
from pdf_rag import ChatPDF
from csvrag import CSVDataset
from hfdataset import Dataset
from urlrag import URLs
from Gitrepo import GithubRepo
from markdownrag import MarkdownRAG

st.set_page_config(page_title="ChatPDF")

def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner("Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(
            f"Ingesting {file.name}"
        ):
            t0 = time.time()
            st.session_state["assistant"].ingest(file_path)
            t1 = time.time()

        st.session_state["messages"].append(
            (
                f"Ingested {file.name} in {t1 - t0:.2f} seconds",
                False,
            )
        )
        os.remove(file_path)

def initialize_assistant(upload_option):
    # Initialize the correct assistant based on the selected option
    if upload_option == "PDF files":
        return ChatPDF()
    elif upload_option == "CSV files":
        return CSVDataset()
    elif upload_option == "Web and doc URLs":
        return URLs()
    elif upload_option == "Hugging Face datasets":
        return Dataset()
    elif upload_option == "Github Repo Links":
        return GithubRepo()
    elif upload_option == "Markdown":
        return MarkdownRAG()
    else:
        return ChatPDF()  # Default option

def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = initialize_assistant("PDF files")  # Default to PDF

    st.header("Secure offline RAG system")

    # Sidebar with file and input options
    with st.sidebar:
        st.header("Upload Options")
        upload_option = st.selectbox(
            "Choose the type of file to upload or input:",
            ["PDF files", "CSV files", "Github Repo Links", "Markdown", "Web and doc URLs", "Hugging Face datasets"],
            key="upload_option",
        )
        # Update the assistant class if the option changes
        if "last_upload_option" not in st.session_state or st.session_state["last_upload_option"] != upload_option:
            st.session_state["assistant"] = initialize_assistant(upload_option)
            st.session_state["last_upload_option"] = upload_option

    st.subheader("Upload a document or enter data")

    # Handling file uploads and inputs based on selection
    if upload_option in ["PDF files", "CSV files"]:
        file_type = {"PDF files": "pdf", "CSV files": "csv"}[upload_option]
        st.file_uploader(
            f"Upload {file_type.upper()} document",
            type=[file_type],
            key="file_uploader",
            on_change=read_and_save_file,
            label_visibility="collapsed",
            accept_multiple_files=True,
        )
    elif upload_option == "Github Repo Links":
        repo_link = st.text_input("Enter the GitHub repository link", key="github_repo_link")
        if repo_link:
            st.session_state["assistant"].ingest(repo_link)
            st.session_state["messages"].append((f"Ingested GitHub repository: {repo_link}", False))

    elif upload_option == "Web and doc URLs":
        web_url = st.text_input("Enter web or document URL", key="web_url")
        if web_url:
            st.session_state["assistant"].ingest(web_url)
            st.session_state["messages"].append((f"Ingested URL: {web_url}", False))
    
    elif upload_option == "Hugging Face datasets":
        dataset_name = st.text_input("Enter Hugging Face dataset name", key="dataset_name")
        if dataset_name:
            st.session_state["assistant"].ingest(dataset_name)
            st.session_state["messages"].append((f"Ingested Hugging Face dataset: {dataset_name}", False))

    elif upload_option == "Markdown":
        file_type = {"Markdown": "md"}[upload_option]
        st.file_uploader(
            f"Upload {file_type.upper()} document",
            type=[file_type],
            key="file_uploader",
            on_change=read_and_save_file,
            label_visibility="collapsed",
            accept_multiple_files=True,
        )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

if __name__ == "__main__":
    page()