#***********************************************************************
#      This file creates the vector database with Chroma               #
#                         Author: Sujay Ray                            #
#                        Date: 19th Nov 2025                           #
#***********************************************************************

from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from chromadb.utils import embedding_functions
import uuid
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
#--------------------------- Chroma embedding function wrapper ------------------------
class ChromaEmbeddingFunction:
    def __init__(self, model_name):
        # Initialize the HuggingFace embeddings model with the given name
        self.hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self._model_name = model_name  # store name for the .name() method

    def __call__(self, *, input):
        # Chroma calls this to get embeddings
        return self.hf_embeddings.embed_documents(input)

    def name(self):
        # Return the model name for Chroma
        return self._model_name

#-------------------------------------------------------------------------------------
CHROMA_PATH = sys.argv[1]
EMBEDDING_NAME = sys.argv[2]
DOCS_PATH = sys.argv[3]

EMBEDDING_MODEL_NAME = EMBEDDING_NAME
# if EMBEDDING_MODEL_NAME == "dangvantuan/french-document-embedding":
#     embeddings = STLangChainWrapper(EMBEDDING_MODEL_NAME)
# else:
#     embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
embeddings = ChromaEmbeddingFunction(EMBEDDING_MODEL_NAME)

if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)
os.makedirs(CHROMA_PATH, exist_ok=True)

client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

collection = client.get_or_create_collection(
    name="collection_test_0",
    embedding_function=embeddings,
    metadata={"hnsw:space": "cosine"}
)

documents = []
metadatas = []
ids = []

with open(DOCS_PATH, "rb") as f:
    chunks = pickle.load(f)

for chunk in chunks:
    documents.append(chunk.page_content)
    metadatas.append(chunk.metadata)
    ids.append(str(uuid.uuid4()))

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)






# with open(DOCS_PATH, "rb") as f:
#     chunks = pickle.load(f)

# if os.path.exists(CHROMA_PATH):
#     shutil.rmtree(CHROMA_PATH)
# os.makedirs(CHROMA_PATH, exist_ok=True)

# vectorstore = Chroma.from_documents(
#     documents=chunks,
#     embedding=embeddings,
#     persist_directory=CHROMA_PATH,
#     #collection_name="my_collection"
# )
# vectorstore.persist()