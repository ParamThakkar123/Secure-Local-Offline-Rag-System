# 🛡️ **Secure Offline RAG System**

We built a RAG system which runs locally on cpu in an offline mode. It uses open source large language models for performing retrieval augmented generation. 

## 🚀 **Tech Stack**
### **Programming Language**
#### 🐍 Python
### **Frameworks & Libraries**
#### 🎨 Streamlit — For building the interactive and intuitive user interface.
#### 🔗 Langchain — To streamline and optimize the RAG pipeline.
#### 🧠 Ollama — Efficient, local LLM deployment for high-quality inference.
#### 🤗 Hugging Face — Powerful models and tools for natural language processing.
### **Vector Database**
#### 🔍 FAISS (Facebook AI Similarity Search) — Fast, efficient, and scalable vector search for document retrieval.
### **Reranking Model**
#### 🎯 BAAI/bge-reranker-base — Advanced model for reranking results to ensure relevant and accurate information is returned.

## ✨ **Features** 

- Minimum CPU memory and RAM usage
- Runs locally even in an offline environment (For PDFs and other documents)
- Highly efficient and quantized model
- Multilingual support with over 29 languages including Chinese
- Fast inference
- Intuitive UI
- Add new documents to the system without the need for a complete reindexing process, ensuring dynamic and flexible integration of new knowledge.
 - Built with a focus on minimizing memory usage, the system leverages lightweight retrieval techniques such as FAISS (or alternatives like inverted indices) to manage large datasets without consuming excessive memory.
- Low Latency

## 📂 **File Structure**
![Screenshot 2024-11-15 150722](https://github.com/user-attachments/assets/ad5bf8bf-634a-477d-aee0-d304026b69ae)


## 🛠️ **Installation Steps**

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
## 📸 **Output Screenshots**
![Screenshot 2024-11-15 170214](https://github.com/user-attachments/assets/14558b74-f65b-460f-9ec7-1e16b6bf52b9)


![Screenshot 2024-11-15 182035](https://github.com/user-attachments/assets/6d45e6f0-cd23-47c3-87d8-7ea1aaf3de85)

## 🎥 **Demo Video**
https://github.com/user-attachments/assets/b825dcbd-7965-4ba1-ae37-a6f0fce77aad

