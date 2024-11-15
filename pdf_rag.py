from langchain_core.globals import set_verbose, set_debug
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
import os

set_debug(True)
set_verbose(True)

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, llm_model: str = "qwen2:0.5b-instruct-q3_K_S"):
        self.model = ChatOllama(model=llm_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=50
        )
        self.prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    "You are a helpful assistant that can answer questions about the PDF document that uploaded by the user. ",
                ),
                (
                    "human",
                    "Here is the document pieces: {context}\nQuestion: {question}",
                ),
            ]
        )
        
        # Initialize HuggingFace embeddings with the specified model
        self.embedding =  OllamaEmbeddings(model="nextfire/paraphrase-multilingual-minilm:l12-v2")

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embedding
        )
        
        # Save the FAISS index
        self.vector_store.save_local("faiss_index")

    def ask(self, query: str):
        if not self.vector_store and os.path.exists("faiss_index"):
            self.vector_store = FAISS.load_local(
                "faiss_index",
                self.embedding,
                allow_dangerous_deserialization=True
            )

        if not self.vector_store:
            return "Please, add a PDF document first."

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.0},
        )

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
        # Remove saved FAISS index
        if os.path.exists("faiss_index"):
            import shutil
            shutil.rmtree("faiss_index")     