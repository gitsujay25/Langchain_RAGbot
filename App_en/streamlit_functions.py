#***********************************************************************
#    This file contains all the functions for streamlit app body       #
#                         Author: Sujay Ray                            #
#                        Date: 19th Nov 2025                           #
#***********************************************************************

import streamlit as st
import utils as ult
import time
import os
import re
import json

#------------------------------ Mapping of parameters ----------------------
role_map = {
    "human": "user",
    "ai": "assistant"
}

model_map = {
    "Llama3 quantized" : "llama3:8b-instruct-q4_K_M",
    "Llama3" : "lamma3",
    "Mistral" : "mistral"
}

embedding_map = {
    "Paraphrase": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "Multilingual-e5" : "intfloat/multilingual-e5-large",
}
#------------------------------ Mapping functions ----------------------
def get_modelmap(model):
    return model_map.get(model)

def get_embeddingmap(model):
    return embedding_map.get(model)
#------------------------------ file_uploader ----------------------
# This function uploads the documents in the DOCUMENTS_FOLDER and
# lists in the sidebar
def file_uploader(DOCUMENTS_FOLDER):
    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True) # Create Documents folder if missing

    with st.sidebar:
        st.header("Upload Documents")
        with st.form("upload_form", clear_on_submit=True):
            uploaded_files = st.file_uploader(
                "Select files to upload",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            submit = st.form_submit_button("Upload")
        
        if submit and uploaded_files:
            for file in uploaded_files:
                save_path = os.path.join(DOCUMENTS_FOLDER, file.name)
                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())
            st.success("Files uploaded successfully!")

        st.header("Uploaded Documents")
        files = os.listdir(DOCUMENTS_FOLDER)
        with st.container(height=130, border=True): # Adding a border to help visualize the container
            if files:
                for f in files:
                    st.write(f)  # just use st.write
            else:
                st.write("No files uploaded yet.")
    return files
#------------------------------ setup_rag_pipeline ----------------------
# This function sets up the main RAG: it takes the files from the doc_folder
# and splits the documents into chunks (see process_documents function in ult.py
# for details) and creates a vectordatabase in chroma_path with the embedding
# model embedding_name. Then it creates the RAG chain with LLM model model_name.
def setup_rag_pipeline(doc_folder, files, chroma_path, model_name, embedding_name):
    documents=[]
    if not files:
            raise FileNotFoundError(f"There are no files in the folder {doc_folder}")
    #st.write("--- Running expensive RAG setup function (only once!) ---")
    for FILE_NAME in files:
        FILE_PATH = os.path.join(doc_folder, FILE_NAME)

        if not os.path.exists(FILE_PATH):
            raise FileNotFoundError(f"The file {FILE_NAME} cannot be found in the folder {doc_folder}")
        
        document = ult.process_documents(FILE_PATH)
        documents.extend(document)
    ult.build_vectorstore_subprocess(documents, chroma_path, embedding_name)
    retriever = ult.get_retriever_from_vectorstore(chroma_path, embedding_name)
    conversational_rag_chain, doc_retriever = ult.get_conversational_rag_chain(retriever, model_name) #doc_retriever
    return conversational_rag_chain, doc_retriever
#------------------------------ build_session_state ----------------------
# This function builds all the session states if not present
def build_session_state(conversational_rag_chain, doc_retriever, model_name, embedding_name):
    st.session_state.rag = conversational_rag_chain 
    st.session_state.ret = doc_retriever
    st.session_state.model = model_name
    st.session_state.embedding = embedding_name
#------------------------------ show_previous_messages ----------------------
# This function shows all the previous chat messages in the chatbox
def show_previous_messages():
    for msg in st.session_state.chat_history:
        role = role_map.get(msg.type, "assistant")
        with st.chat_message(role):
            st.write(msg.content)
#------------------------------ stream_ai_message ----------------------
# This function streams the LLM answer to the queries named user_input
def stream_ai_message(user_input,conversational_rag_chain):
    # 1. Show the user message -------------------------
    with st.chat_message("user"):
        st.write(user_input)
    # 1. Show the AI message -------------------------
    with st.chat_message("assistant"):
        full_response_text = "" # initialize the response as blank
        response_placeholder = st.empty() # Create a placeholder in the UI for the dynamic output

        #start = time.time()
        #  Iterate through the stream
        for chunk in conversational_rag_chain.stream({
            "input": user_input,
            "chat_history": st.session_state.chat_history
        }):
            # The 'chunk' might be a dictionary or object if your chain is complex. 
            # Extract the relevant text part:
            # If using LangChain LCEL, 'chunk' often contains the 'answer' or 'response' key
            ai_response_chunk = chunk.get("answer", "") if isinstance(chunk, dict) else chunk
            full_response_text += ai_response_chunk
            # Update the UI instantly with the accumulated text
            response_placeholder.markdown(full_response_text)
        #end_time = time.time()
        #total_duration = end_time - start
        #st.write(f"Total time taken: {total_duration:.2f} seconds")

        return full_response_text
#------------------------------ stop words ----------------------
def load_stopwords(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        stopwords = json.load(f)

    return set(word.lower() for word in stopwords)
#------------------------------ extract_keywords ----------------------
def extract_keywords(query: str) -> list[str]:
    words = re.findall(r"\b\w+\b", query.lower())
    #STOPWORDS = load_stopwords("stop_words_english.json")
    STOPWORDS = load_stopwords("stop_words_english.json")

    keywords = [
        re.escape(word)
        for word in words
        if len(word) > 2 and word not in STOPWORDS
    ]

    return keywords
#------------------------------ build_highlight_pattern ----------------------
def build_highlight_pattern(query: str):
    keywords = extract_keywords(query)

    if not keywords:
        return None

    return re.compile(
        rf"\b({'|'.join(keywords)})\b",
        re.IGNORECASE
    )
#------------------------------ build_keyword_pattern ----------------------
def build_keyword_pattern(keywords: list[str]):
    expanded_keywords = [
        rf"\w*{re.escape(word)}\w*" for word in keywords
    ]

    return re.compile(
        rf"\b({'|'.join(expanded_keywords)})\b",
        re.IGNORECASE
    )
#------------------------------ highlight_sentences ----------------------
def highlight_sentences(text: str, query: str) -> str:
    keywords = extract_keywords(query)

    if not keywords:
        return text

    # Split text into sentences (basic but effective)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    keyword_pattern = build_keyword_pattern(keywords)

    highlighted_sentences = []
    # keyword_pattern = re.compile(
    #     rf"\b({'|'.join(map(re.escape, keywords))})\b",
    #     re.IGNORECASE
    # )

    for sentence in sentences:
        if keyword_pattern.search(sentence):
            sentence = (
                "<span style='background-color:#ffeb99;"
                "padding:2px 4px; border-radius:4px;'>"
                f"{sentence}</span>"
            )
        highlighted_sentences.append(sentence)

    return " ".join(highlighted_sentences)
#------------------------------ highlight text ----------------------
# This function gives highlight texts from text which appears in query
# def highlight_text(text: str, query: str) -> str:
#     """
#     Highlights words from query that appear in text.
#     """
#     if not query:
#         return text

#     # Split query into words and remove short/common tokens
#     keywords = [re.escape(word) for word in query.split() if len(word) > 2]

#     if not keywords:
#         return text

#     # Create regex pattern (case-insensitive)
#     pattern = re.compile(rf"({'|'.join(keywords)})", re.IGNORECASE)

#     # Replace matches with highlighted span
#     highlighted_text = pattern.sub(
#         r"<span style='background-color:#ffeb99; padding:2px 4px; border-radius:4px; font-weight:600;'>\1</span>",
#         text
#     )

#     return highlighted_text
#------------------------------ show_context ----------------------
# This function shows the relevant contexts to the query user_input
# using the retriever doc_retriever
def show_context(doc_retriever, user_input):
    docs = doc_retriever.invoke({
        "input": user_input,
        "chat_history": st.session_state.chat_history
    })
    for i,d in enumerate(docs,1):
        # pattern = build_highlight_pattern(user_input)
        # if pattern:
        #     highlighted_text = pattern.sub(
        #         r"<span style='background-color:#fff3cd; font-weight:600;'>\1</span>",
        #         d.page_content
        #     )
        # else:
        #     highlighted_text = d.page_content
        # highlighted_content = highlight_text(d.page_content, user_input)
        highlighted_content = highlight_sentences(d.page_content,user_input)
        st.markdown(
            f"""
            <p style='font-size:15px; font-weight:400; color:#0000FF; text-align:justify; text-align-last:center; width:100%;'>
                <span style='font-weight:700; color:#CC0000;'>Relevant text {i}:</span><br>
                <span style='font-weight:600; color:#CC0000;'>Source:</span> {d.metadata.get("source").split("/")[-1]}, &nbsp;
                <span style='font-weight:600; color:#CC0000;'>Page:</span> {d.metadata.get("page_label")}
            </p>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <p style='font-size:14px; font-weight:400; color:#595959;
                    text-align:justify; text-align-last:left; width:100%;'>
                {highlighted_content}
            </p>
            """,
            unsafe_allow_html=True
        )
        # st.markdown(
        #     f"""
        #     <p style='font-size:14px; font-weight:400; color:#595959; text-align:justify; text-align-last:left; width:100%;'>
        #         {d.page_content}
        #     </p>
        #     """,
        #     unsafe_allow_html=True
        # )
#------------------------------------------------------------------------------------------------------------------------------------------------------------