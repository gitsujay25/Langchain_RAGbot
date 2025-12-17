#*********************************************************************************#
#    Projet Chatbot RAG : Langchain RAG chatbot avec récupération de contexte     #
#                         Auteur: Sujay Ray                                       #
#                        Date: 15 décembre 2025                                   #
#*********************************************************************************#

import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage
import streamlit_functions as stfnc

#------------------------------ page layout ----------------------
st.set_page_config(layout="wide")
middle, bfr2, left = st.columns([2.3,0.2,2])
#------------------------------- Relevant folder paths ----------------------------------------------------------------
os.makedirs("Documents", exist_ok=True)
os.makedirs("Chroma_store", exist_ok=True)
DOCUMENT_FOLDER_PATH = "Documents"
CHROMA_PATH = "./Chroma_store"
#---------------------------- side bar -------------------------------------------------------------------------------
model_names = ["Llama3 quantized","Llama3","Mistral"]
embedding_names = ["Paraphrase","Multilingual-e5","French embedding"]
st.sidebar.header("Configurer le LLM et l’Embedding")
model = st.sidebar.selectbox('Sélectionner un modèle LLM', model_names, key="select_model")
MODEL_NAME = stfnc.get_modelmap(model)
embedding = st.sidebar.selectbox('Sélectionner un modèle d’embedding', embedding_names, key="select_embedding")
EMBEDDING_NAME = stfnc.get_embeddingmap(embedding)
FILES = stfnc.file_uploader(DOCUMENT_FOLDER_PATH)

#------------------------------------ checck session_state and initialize ----------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag" not in st.session_state:
    st.session_state.rag = None
if "ret" not in st.session_state:
    st.session_state.ret = None
if "model" not in st.session_state:
    st.session_state.model = "llama3:8b-instruct-q4_K_M" # default
if "embedding" not in st.session_state:
    st.session_state.embedding = "intfloat/multilingual-e5-large" # default

#----------------------------------------- Define and build the ragchain ---------------------------------------------------
if st.session_state.rag is None or st.session_state.ret is None:
    if st.sidebar.button("Créer le RAGbot"):
        with st.sidebar:
            with st.spinner("Création du RAGbot, veuillez patienter..."):
                conversational_rag_chain, doc_retriever = stfnc.setup_rag_pipeline(DOCUMENT_FOLDER_PATH,FILES, CHROMA_PATH, MODEL_NAME, EMBEDDING_NAME)
                stfnc.build_session_state(conversational_rag_chain, doc_retriever, MODEL_NAME, EMBEDDING_NAME)
                st.success("RAGbot prêt à l'emploi!")
    else:
        st.warning("Créez d'abord votre RAGbot pour discuter!")
        st.stop()
elif (st.session_state.model != MODEL_NAME) or (st.session_state.embedding != EMBEDDING_NAME):
    with st.sidebar:
        st.info("Créer le RAGbot avec votre nouveau modèle")
        if st.button("Créer le RAGbot"):
            with st.spinner("Création du RAGbot, veuillez patienter..."):
                st.session_state.chat_history = []
                conversational_rag_chain, doc_retriever = stfnc.setup_rag_pipeline(DOCUMENT_FOLDER_PATH,FILES, CHROMA_PATH, MODEL_NAME, EMBEDDING_NAME)
                stfnc.build_session_state(conversational_rag_chain, doc_retriever, MODEL_NAME, EMBEDDING_NAME)
                st.success("RAGbot prêt à l'emploi!")
else:
    with st.sidebar:
        if st.button("Recréer le RAGbot"):
            with st.spinner("Création du RAGbot, veuillez patienter..."):
                st.session_state.chat_history = []
                conversational_rag_chain, doc_retriever = stfnc.setup_rag_pipeline(DOCUMENT_FOLDER_PATH,FILES, CHROMA_PATH, MODEL_NAME, EMBEDDING_NAME)
                stfnc.build_session_state(conversational_rag_chain, doc_retriever, MODEL_NAME, EMBEDDING_NAME)
                st.success("RAGbot prêt à l'emploi!")
        else:
            st.info("RAGbot est en action. Appuyez sur «Recréer le RAGbot» pour le relancer")
st.write(st.session_state.model)
st.write(st.session_state.embedding)
conversational_rag_chain = st.session_state.rag
doc_retriever = st.session_state.ret
#----------------------------------------- Display existing messages ---------------------------------------------------
with middle:
    st.markdown(
        """
        <p style='font-size:25px; font-weight:800; color:black; text-align:justify; text-align-last:center; width:100%;'>
            Langchain RAGBot
        </p>
        """,
        unsafe_allow_html=True
    )
    chat_container = st.container(height=600, border=True)
    with chat_container:
        stfnc.show_previous_messages()
#----------------------------------------- Take user query ---------------------------------------------------
with middle:
    user_input = st.chat_input("Posez une question sur vos documents")
#----------------------------------------- The main llm part ---------------------------------------------------
if user_input :
    with middle:
        with chat_container:
            full_response_text = stfnc.stream_ai_message(user_input,conversational_rag_chain)
    #----------------------------------------- The context part ---------------------------------------------------
    with left:
        st.markdown(
            """
            <p style='font-size:20px; font-weight:600; color:black; text-align:justify; text-align-last:center; width:100%;'>
                Envie d’en savoir plus ou de vérifier la réponse !<br> Consultez les textes pertinents.
            </p>
            """,
            unsafe_allow_html=True
        )
        chat_container_left = st.container(height=650, border=True)
        with chat_container_left:
            stfnc.show_context(doc_retriever, user_input)
    #----------------------------------------- update chat history ---------------------------------------------------                
    ai_response = full_response_text
    st.session_state.chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=ai_response)
    ])