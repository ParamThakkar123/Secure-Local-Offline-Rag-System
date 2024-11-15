from langchain_core.globals import set_verbose, set_debug
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import pandas as pd
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain_ollama import OllamaEmbeddings
import os

set_debug(True)
set_verbose(True)

class CSVDataset:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, llm_model: str = "qwen2:0.5b-instruct-q3_K_S"):
        self.model = ChatOllama(model=llm_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, 
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]  # Optimized for CSV row structure
        )
        
        # CSV-specific prompt template
        self.prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    """You are a helpful assistant that answers questions about CSV data. 
                    The context provided contains rows from a CSV file where each row represents a record.
                    Analyze the data carefully and provide accurate, data-driven responses.
                    If asked about calculations or trends, explain your reasoning."""
                ),
                (
                    "human",
                    """Context from the CSV data:
                    {context}
                    
                    Question: {question}
                    
                    Please provide a clear answer based on the CSV data. If you need to make calculations or identify patterns, 
                    explain your methodology."""
                ),
            ]
        )

        # Initialize embeddings
        self.embedding = OllamaEmbeddings(
            model="nextfire/paraphrase-multilingual-minilm:l12-v2"
        )
        
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        self.compressor = CrossEncoderReranker(model=model, top_n=3)

    def ingest(self, dataset_path: str, csv_args: dict = None):
        """
        Ingest CSV data with custom handling options.
        
        Args:
            dataset_path: Path to the CSV file
            csv_args: Optional dictionary of pandas read_csv arguments
        
        Raises:
            RuntimeError: If there's an error loading or processing the CSV file
        """
        try:
            if csv_args is None:
                csv_args = {
                    'encoding': 'utf-8',
                    'on_bad_lines': 'skip',  # Skip problematic rows
                    'engine': 'python'  # More flexible engine for handling various CSV formats
                }
            
            # First read with pandas to get column info
            try:
                df = pd.read_csv(dataset_path, **csv_args)
            except UnicodeDecodeError:
                # Try with different encoding if UTF-8 fails
                csv_args['encoding'] = 'latin-1'
                df = pd.read_csv(dataset_path, **csv_args)
                
            column_names = df.columns.tolist()
            
            # Convert DataFrame to documents
            documents = []
            for index, row in df.iterrows():
                # Convert row to string representation
                content = "\n".join([f"{col}: {row[col]}" for col in column_names])
                metadata = {
                    "source": dataset_path,
                    "row_index": index,
                    "column_names": column_names,
                    "total_rows": len(df)
                }
                documents.append(Document(page_content=content, metadata=metadata))
            
            # Split documents
            chunks = self.text_splitter.split_documents(documents)
            
            # Create vector store
            self.vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=self.embedding
            )
            
            # Save the FAISS index and metadata
            os.makedirs("faiss_index", exist_ok=True)
            self.vector_store.save_local("faiss_index")
            
            # Save column names for later use
            with open("csv_metadata.txt", "w", encoding='utf-8') as f:
                f.write(",".join(column_names))
                
            return f"Successfully processed CSV file with {len(df)} rows and {len(column_names)} columns."
            
        except Exception as e:
            raise RuntimeError(f"Error processing CSV file {dataset_path}: {str(e)}")

    def ask(self, query: str, k: int = 5):
        """
        Query the CSV dataset with improved retrieval and response generation.
        
        Args:
            query: The question to ask about the CSV data
            k: Number of relevant chunks to retrieve
        """
        try:
            if not self.vector_store and os.path.exists("faiss_index"):
                self.vector_store = FAISS.load_local(
                    "faiss_index",
                    self.embedding,
                    allow_dangerous_deserialization=True
                )

            if not self.vector_store:
                return "Please add a CSV dataset first."

            # Create retriever with compression pipeline
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": k,
                    "score_threshold": 0.5,  # Adjusted for CSV data
                }
            )

            # Create compression pipeline
            self.compressed_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor,
                base_retriever=self.retriever    
            )

            # Enhanced chain with metadata handling
            self.chain = (
                {
                    "context": self.compressed_retriever,
                    "question": RunnablePassthrough()
                }
                | self.prompt
                | self.model
                | StrOutputParser()
            )

            return self.chain.invoke(query)
            
        except Exception as e:
            return f"Error processing query: {str(e)}"

    def clear(self):
        """Clear all stored data and indices."""
        self.vector_store = None
        self.retriever = None
        self.chain = None
        
        # Remove saved files
        if os.path.exists("faiss_index"):
            import shutil
            shutil.rmtree("faiss_index")
        if os.path.exists("csv_metadata.txt"):
            os.remove("csv_metadata.txt")