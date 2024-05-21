import logging
from dotenv import load_dotenv

from faiss_client import FAISS_CLIENT

DOC_DIR="D:\\000.Dev\\open-webui\\backend\\data"
VECTOR_DIR="D:\\000.Dev\\open-webui\\backend\\data\\vector_db_faiss"
LLM_MODEL="EEVE-Korean-10.8B-Q4:latest"
EMBEDDING_MODEL="nomic-embed-text:latest"
OLLAMA_URL="http://localhost:11434"

load_dotenv()
logging.basicConfig()
logging.getLogger("faiss_client").setLevel(logging.DEBUG)


faiss_client = FAISS_CLIENT(VECTOR_DIR, LLM_MODEL,EMBEDDING_MODEL,OLLAMA_URL)
print(f"{faiss_client.get_collections()}")

faiss_client.create_collection("./data/회사사규(2023.11) - 복사본.pdf")

docs = faiss_client.search("휴일 및 휴가","all")
for doc in docs:
    print("------- after faiss_client.search -----------------")
    print(doc)


