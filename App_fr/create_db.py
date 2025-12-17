#***********************************************************************
#      This file creates the vector database with Chroma               #
#                         Author: Sujay Ray                            #
#                        Date: 19th Nov 2025                           #
#***********************************************************************

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import pickle,sys,os,shutil

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
#-------------------------------------------------------------------------------------
CHROMA_PATH = sys.argv[1]
EMBEDDING_NAME = sys.argv[2]
DOCS_PATH = sys.argv[3]

EMBEDDING_MODEL_NAME = EMBEDDING_NAME
if EMBEDDING_MODEL_NAME == "dangvantuan/french-document-embedding":
    embeddings = STLangChainWrapper(EMBEDDING_MODEL_NAME)
else:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

with open(DOCS_PATH, "rb") as f:
    chunks = pickle.load(f)

if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)
os.makedirs(CHROMA_PATH, exist_ok=True)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH,
    #collection_name="my_collection"
)
vectorstore.persist()