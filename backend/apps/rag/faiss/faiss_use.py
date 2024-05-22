import logging
from dotenv import load_dotenv

from faiss_client import FAISS_CLIENT

DOC_DIR="D:\\000.Dev\\open-webui\\backend\\data"
VECTOR_DIR="D:\\000.Dev\\open-webui\\backend\\data\\vector_db_faiss"
VECTOR_DIR_OPENAI="D:\\000.Dev\\open-webui\\backend\\data\\vector_db_faiss_openai"
VECTOR_DIR_HF="D:\\000.Dev\\open-webui\\backend\\data\\vector_db_faiss_hf"
LLM_MODEL="EEVE-Korean-10.8B-Q4:latest"
EMBEDDING_MODEL="nomic-embed-text:latest"
OLLAMA_URL="http://localhost:11434"

load_dotenv()
logging.basicConfig()
logging.getLogger("faiss_client").setLevel(logging.INFO)


faiss_client = FAISS_CLIENT(VECTOR_DIR, LLM_MODEL,EMBEDDING_MODEL,OLLAMA_URL,type="ollama")
faiss_client_openai = FAISS_CLIENT(VECTOR_DIR_OPENAI, LLM_MODEL,EMBEDDING_MODEL,OLLAMA_URL,type="openai")
faiss_client_hf = FAISS_CLIENT(VECTOR_DIR_HF, LLM_MODEL,EMBEDDING_MODEL,OLLAMA_URL,type="huggingface")
print(f"ollama -> {faiss_client.get_collections()}")
print(f"openai -> {faiss_client_openai.get_collections()}")
print(f"hf -> {faiss_client_hf.get_collections()}")

faiss_client.create_collection("./data/회사사규(2023.11) - 복사본.pdf")
faiss_client_openai.create_collection("./data/회사사규(2023.11) - 복사본.pdf")
faiss_client_hf.create_collection("./data/회사사규(2023.11) - 복사본.pdf")

# query = "휴일 및 휴가에 대한 규정은?"
# query = "휴직이나 퇴직 규정은?"
query = "경조사 휴가 경조사 지원금"

docs = faiss_client.search(query,"all")
for doc in docs:
    print("------- after faiss_client.search -----------------")
    print(doc)

docs_openai = faiss_client_openai.search(query,"all")
for doc in docs_openai:
    print("------- after faiss_openai_client.search -----------------")
    print(doc)

docs_hf = faiss_client_hf.search(query,"all")
for doc in docs_hf:
    print("------- after faiss_huggingface_client.search -----------------")
    print(doc)

print("\n\n\n\n")
docs = faiss_client.search_llm(query,"all")
for doc in docs:
    print("------- after faiss_client.search_llm -----------------")
    print(doc)

docs_openai = faiss_client_openai.search_llm(query,"all")
for doc in docs_openai:
    print("------- after faiss_openai_client.search_llm -----------------")
    print(doc)

docs_hf = faiss_client_hf.search_llm(query,"all")
for doc in docs_hf:
    print("------- after faiss_huggingface_client.search_llm -----------------")
    print(doc)
