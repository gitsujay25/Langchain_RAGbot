#***********************************************************************
#    This file contains all the relevant utility function for the RAG  #
#                         Author: Sujay Ray                            #
#                        Date: 19th Nov 2025                           #
#***********************************************************************

import subprocess, pickle, time
import numpy as np
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

#---------- Langchain Wrapper for embedding model of type SentenceTransformer ---------
class STLangChainWrapper(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True
        )

    def embed_documents(self, texts):
        return self.model.encode(
            texts,
            show_progress_bar=True
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()
#------------------------------------- process_documents -------------------------------------
# This function takes pdf files from file_path and splits the documents into chunks of size
# chunk_size with overlap of size chunk_overlap. Then it returns the chunks
def process_documents(file_path):
    print(f"Loading documents from {file_path}...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        separators=[
            "\n\n",      # paragraphs
            "\n",        # line breaks
            ". ",        # sentence end (English / French)
            "? ",
            "! ",
            "; ",
            ", ",
            " "          # word boundary (last resort)
        ],
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
#-------------------------- build_vectorstore_subprocess -----------------------------
# This function builds the vector database using subprocess. It takes the chunks and
# and builds the vector database of the chunks in CHROMA_PATH using the embedding
# model named EMBEDDING_NAME
def build_vectorstore_subprocess(chunks, CHROMA_PATH, EMBEDDING_NAME, DOCS_PATH="./chunks.pkl"):
    # Save chunks to disk so subprocess can read
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    subprocess.run(["python", "create_db.py", CHROMA_PATH, EMBEDDING_NAME, DOCS_PATH], check=True)
    time.sleep(15)

#-------------------------- get_retriever_from_vectorstore -----------------------------
# This function builds a retriever from the vector database at CHROMA_PATH
def get_retriever_from_vectorstore(CHROMA_PATH, EMBEDDING_MODEL_NAME):
    #EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    if EMBEDDING_MODEL_NAME == "dangvantuan/french-document-embedding":
        embeddings = STLangChainWrapper(EMBEDDING_MODEL_NAME)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    COLLECTON_NAME = "collection_test_0"
    vectorstore = Chroma(
        collection_name=COLLECTON_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    # vectorstore = Chroma(
    #     persist_directory=CHROMA_PATH,
    #     embedding_function=embeddings,
    #     #collection_name="default"
    # )
    print("Embedding used:", embeddings)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "lambda_mult": 0.8}
    )
#-------------------------- get_conversational_rag_chain -----------------------------
# This function builds the main RAG using LLM model MODEL_NAME and the retriever
def get_conversational_rag_chain(retriever, MODEL_NAME):
    system_prompt = (
        "You are an assistant for question-answering tasks."
        "Use ONLY the following retrieved context to answer the question concisely."
        "If you do not know the answer, simply say that you do not know."
        "Maintain a conversational tone and use the chat history to provide a smooth conversation."
        "\n\n"
        "Context: {context}"
    )

    # "Given the user query and the prior conversation, rewrite the query so it can be answered using a retrieval system. "
    # "Include all relevant context from the conversation. "
    # "Do not introduce any information not mentioned. "
    # "Return only the rewritten query."

    qa_contex_prompt = (
        "Taking into account the user’s query and the previous conversation,"
        "rewrite the query to a stand-alone query so that it can be used effectively by a retrieval system."
        "Include only the relevant context from the conversation if any."
        "Do not add any new information."
        "Return only the rewritten query, without any additional explanation."
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    llm = ChatOllama(model=MODEL_NAME, temperature=0.2, keep_alive="10m")

    output_parser = StrOutputParser()
    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt, output_parser=output_parser)

    history_aware_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_contex_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, history_aware_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)
    return rag_chain, history_aware_retriever
#------------------------------------------------------------------------------------------------------
# def get_retriever(chunks, CHROMA_PATH):
#     #print(f"Creating embeddings and vector store...")
#     EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
#     embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

#     vectorstore = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         persist_directory=CHROMA_PATH
#     )
#     #st.write('Number of documents in vectorstore:', len(vectorstore._collection.get()))
#     return vectorstore.as_retriever(
#         search_type="mmr",
#         search_kwargs={"k": 8, "lambda_mult": 0.8}
#     )

# def get_similarity_score(query_text: str, doc_text: str) -> float:
#     EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
#     embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
#     # Embed both query and document
#     query_emb = embeddings.embed_query(query_text)
#     doc_emb = embeddings.embed_query(doc_text)
    
#     a = np.array(query_emb)
#     b = np.array(doc_emb)
#     score = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
#     # Compute cosine similarity
#     #score = cosine_similarity(query_emb, doc_emb)
#     return float(score)