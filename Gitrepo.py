from utils import clone_repository, load_repo_files
from langchain_core.globals import set_verbose, set_debug
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.document_loaders import NotebookLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
import os
import shutil

set_verbose(True)
set_debug(True)

class GithubRepo:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, llm_model: str = "qwen2:0.5b-instruct-q3_K_S"):
        # Initialize model and text splitter
        self.model = ChatOllama(model=llm_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100
        )
        
        # Define the prompt for the chatbot
        self.prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    "You are a helpful assistant that can answer questions about the PDF document uploaded by the user.",
                ),
                (
                    "human",
                    "Here is the document pieces: {context}\nQuestion: {question}",
                ),
            ]
        )

        # Initialize HuggingFace embeddings
        self.embedding = OllamaEmbeddings(model="nextfire/paraphrase-multilingual-minilm:l12-v2")  

    def ingest(self, url: str):
        # Extract the repository name from the URL
        repo_name = url.split("/")[-1].replace(".git", "")
        self.clone_dir = f'./{repo_name}'
        
        # Ensure the clone directory exists or create it
        if not os.path.exists(self.clone_dir):
            os.makedirs(self.clone_dir)

        # Clone the repository
        clone_repository(url, self.clone_dir)
        
        # Verify if the repo exists after cloning
        if not os.path.isdir(self.clone_dir):
            print("Repository clone failed or directory doesn't exist.")
            return "Failed to clone repository. Please check the URL and try again."
        
        # Load files from the cloned repository
        files = load_repo_files(self.clone_dir)
        if not files:
            print("No files loaded from the repository.")
            return "Failed to load files from the repository."
        
        # Convert files to Document objects
        documents = []
        for file_content in files:
            # Assuming file_content is a string of the file's content
            doc = Document(page_content=file_content)
            documents.append(doc)
        
        # Check for notebook files and load their content
        notebook_files = [f for f in os.listdir(self.clone_dir) if f.endswith(".ipynb")]
        for notebook_file in notebook_files:
            notebook_loader = NotebookLoader(self.clone_dir + "/" + notebook_file)
            notebook_docs = notebook_loader.load()
            documents.extend(notebook_docs)

        # Split the documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        chunks = filter_complex_metadata(chunks)

        # Initialize FAISS with the loaded document chunks
        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embedding
        )

        # Save the FAISS index
        self.vector_store.save_local("faiss_index")

    def ask(self, query: str):
        # If FAISS index doesn't exist, load it from the saved file
        if not self.vector_store and os.path.exists("faiss_index"):
            self.vector_store = FAISS.load_local(
                "faiss_index",
                self.embedding,
                allow_dangerous_deserialization=True
            )

        if not self.vector_store:
            return "Please, add a Repo URL first."

        # Set up the retriever for similarity search
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.0},
        )

        # Set up the chain to process the question
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        return self.chain.invoke(query)
    
    def clear(self):
        # Clear the vector store, retriever, and chain
        self.vector_store = None
        self.retriever = None
        self.chain = None
        
        # Remove saved FAISS index if it exists
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")
        
        # Remove the cloned repository folder if it exists
        if os.path.isdir(self.clone_dir):
            shutil.rmtree(self.clone_dir)