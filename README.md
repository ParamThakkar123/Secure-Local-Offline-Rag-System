# Secure Offline RAG system

We built a RAG system which runs locally on cpu in an offline mode. It uses open source large language models for performing retrieval augmented generation. 

# Tech Stack  
- Python
- Frameworks used and technologies used:Streamlit, langchain, Ollama, huggingface 
- Vector database used: FAISS (Facebook AI Similarity Search)
- Reranking models used: BAAI/bge-reranker-base

# Features  

- Minimum CPU memory and RAM usage
- Runs locally even in an offline environment (For PDFs and other documents)
- Highly efficient and quantized model
- Multilingual support with over 29 languages including Chinese
- Fast inference
- Intuitive UI
- Add new documents to the system without the need for a complete reindexing process, ensuring dynamic and flexible integration of new knowledge.
 - Built with a focus on minimizing memory usage, the system leverages lightweight retrieval techniques such as FAISS (or alternatives like inverted indices) to manage large datasets without consuming excessive memory.
- Low Latency

# File Structure 
![Screenshot 2024-11-15 150722](https://github.com/user-attachments/assets/ad5bf8bf-634a-477d-aee0-d304026b69ae)


# Installation steps : 

### Clone the repository: 

```Python
> git clone https://github.com/ParamThakkar123/Secure-Local-Offline-Rag-System.git
```
### Change directory: 
```Python
> cd Secure-Local-Offline-Rag-System
```
### Installl the dependencies : 
```Python
> pip install -r requirements.txt
```
Download Ollama app and run it

### Open command line and type : 
```Python
> ollama pull qwen2:0.5b-instruct-q3_K_S
> ollama pull nextfire/paraphrase-multilingual-minilm:l12-v2
```


### Run the app.py using the following command in the command line
```Python
> streamlit run app.py
```

### If the above command gives the error “streamlit not recognized”, enter the following command
```Python
> python -m streamlit run app.py
```


