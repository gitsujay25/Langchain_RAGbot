# Langchain RAGBot

**Langchain RAGBot** is an interactive **Streamlit** application that enables users to ask questions about their documents using a **Retrieval-Augmented Generation (RAG)** pipeline powered by **LLMs**.

## 📌 Overview  
This is a **Conversational RAG** built with completely opensource LLM models and Embedding models, which can be easily generalized to even powerful models if the
hardware resources are available. The primary focus of this RAG is to focus on the building a RAG application specifically for documents in French language. Although this model can also be used for documents in English.

The app provides:
- Option to upload documents
- A chat-based interface for querying documents
- Model and embedding selection (Open source models from Huggingface and Ollama)
- Transparent access to the retrieved context used to generate answers with highlighted the contexts

---

## 🚀 Features 

### **1. Multiple LLMs**
  - Llama3 (quantized q4_K_M) 8b
  - Llama3 8b
  - Mistral 7b

### **2. Multiple Embeddings**
  - Paraphrase (sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
  - Multilingual-e5 (intfloat/multilingual-e5-large)

### **3. Document Upload**
  - Upload documents and query them instantly

### **4. Interactive Chat UI**
  - Persistent chat history using Streamlit session state

### **5. Streaming Responses**
  - AI answers are streamed token-by-token for better UX

### **6. Context Inspection**
  - View the relevant retrieved document chunks for each question with the most relevant sentences highlighted

---

## 📁 Project Structure  
```text
Langchain_RAGbot/App_en
│
├── app.py                   # Main Streamlit application
├── utils.py                 # Helper functions (to build document uploader, retriever, rag chain etc.)
├── streamlit_functions.py   # Helper functions for streamlit app (show document uploader, chat message etc.)
├── create_db.py             # File for creating chroma database
├── requirement.txt          # File containing the required packages
│
├── Chroma_store/            # Folder containing the vector database
├── Documents/               # Folder containing the document files
│
├── images/                  # Saved figures, icons, screenshots
│   └── example.png
│
└── README.md                # Documentation
```

## 🛠️ Installation

### ⚠️ Prerequisites
Before running the application, make sure the following requirements are met:
- Ollama is installed and running on your system
- The required LLM models are downloaded in Ollama
- The required embedding models are downloaded and available

The application will not function correctly unless these components are installed beforehand.

### 📥 Clone the repository
```bash
git clone https://github.com/gitsujay25/Langchain_RAGbot.git
cd Langchain_RAGbot/App_en
conda create -n langchain_rag python=3.10
conda activate langchain_rag
pip install -r requirements.txt
```

### ▶️ Run the Application

```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at: http://localhost:8501/

## 🏗️ Application Layout

- **Sidebar**
  - LLM model selection
  - Embedding model selection
  - Document uploader
  - Build or rebuild the RAG option

- **Center Panel**
  - Chat interface
  - User input and AI responses

- **Left Panel**
  - Retrieved context related to the user query

---

## ❓ How to use

1. Select an **LLM model** from the sidebar  
2. Select an **embedding model**  
3. Upload one or more **documents**
4. Press the Build RAG button (or Rebuild RAG button - While rebuilding the RAG, all the previous chats will be deleted and the application will start afresh.)
4. Ask a question using the chat input  
5. Receive a **streamed AI response**  
6. Inspect the **retrieved context** in the left panel 

---

## 🧰 Development Tips

- Keep reusable functions in `utils.py`  
- Use `streamlit_functions.py` for streamlit UI building functions   
- Pin versions in `requirements.txt` for reproducibility

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repository, open issues, or submit pull requests.

## 📬 Contact
For questions or suggestions:
- Author: Sujay Ray
- GitHub: https://github.com/gitsujay25
- Linkdin: https://www.linkedin.com/in/sujayray92/