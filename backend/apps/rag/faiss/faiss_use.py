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

# '9163575225d8baded4dcef8a4e71068f553f88613bd3856b8e4546128949687'
faiss_client.create_collection("./data/회사사규(2023.11) - 복사본.pdf")
faiss_client_openai.create_collection("./data/회사사규(2023.11) - 복사본.pdf")
faiss_client_hf.create_collection("./data/회사사규(2023.11) - 복사본.pdf")

#  '22069beb988b6937d65a5fcc993c0b7f4fcd3cf67a319567fc2349da408e6e2'
faiss_client.create_collection("./data/WW챗봇 경조금 및 복리후생비 지급 기준_2024_재무관리실.pdf")
faiss_client_openai.create_collection("./data/WW챗봇 경조금 및 복리후생비 지급 기준_2024_재무관리실.pdf")
faiss_client_hf.create_collection("./data/WW챗봇 경조금 및 복리후생비 지급 기준_2024_재무관리실.pdf")

# 'cb00c3509b954f29582a0edf61a1e70a2b139e66509622e85287bef32b1fd66'
faiss_client.create_collection("./data/WW챗봇 출장여비규정_20240401_재무관리실.pdf")
faiss_client_openai.create_collection("./data/WW챗봇 출장여비규정_20240401_재무관리실.pdf")
faiss_client_hf.create_collection("./data/WW챗봇 출장여비규정_20240401_재무관리실.pdf")

# '508efc958778204b379a3f8b2c3b7ac69bd658e67b8aa9e3c29cdaab523c39b'
faiss_client.create_collection("./data/복귀자 관련 문의 사항.pdf")
faiss_client_openai.create_collection("./data/복귀자 관련 문의 사항.pdf")
faiss_client_hf.create_collection("./data/복귀자 관련 문의 사항.pdf")

# '2ee2ec5a414ae38937393a281cd327345f53364b674dd338a0ebcfd2de0f632'
faiss_client.create_collection("./data/AI시장분석_증권.pdf")
faiss_client_openai.create_collection("./data/AI시장분석_증권.pdf")
faiss_client_hf.create_collection("./data/AI시장분석_증권.pdf")

# collection_name = 'e61cc96091a2fbdb89ffc1832e03946cc0919f0b20de97b264fddf40f70913c'
faiss_client.create_collection("./data/생성AI 확산에 따른 디지털 인재 양성 개선 방안 연구.pdf")
faiss_client_openai.create_collection("./data/생성AI 확산에 따른 디지털 인재 양성 개선 방안 연구.pdf")
faiss_client_hf.create_collection("./data/생성AI 확산에 따른 디지털 인재 양성 개선 방안 연구.pdf")


# query = "휴일 및 휴가에 대한 규정은?"
# query = "휴직이나 퇴직 규정은?"
# query = "경조사 휴가 경조사 지원금"
# query = "프로젝트 복귀자는 어떻게 하면 되나요?" # good quality 복귀자 관련 문의 사항.pdf에서는 최고의 품질
# query = "자녀 출생시 경조금 지급기준은? " # pdf 문서가 짧아서 조회가 잘 되는가? 하여간 good quality
# query = "플리토 주식 성장 가능성 " # 언급된 부분이 일부 조회됨 not good
query = "소프트 스킬에 대한 선호도 " 
collection_name="e61cc96091a2fbdb89ffc1832e03946cc0919f0b20de97b264fddf40f70913c"

docs = faiss_client.search(query,collection_name)
for doc in docs:
    print("------- after faiss_client.search -----------------")
    print(doc)
docs = faiss_client.search_compression(query,collection_name)
for doc in docs:
    print("------- after faiss_client.search_compression -----------------")
    print(doc)

docs_openai = faiss_client_openai.search(query,collection_name)
for doc in docs_openai:
    print("------- after faiss_openai_client.search -----------------")
    print(doc)
docs_openai = faiss_client_openai.search_compression(query,collection_name)
for doc in docs_openai:
    print("------- after faiss_openai_client.search_compression -----------------")
    print(doc)

docs_hf = faiss_client_hf.search(query,collection_name)
for doc in docs_hf:
    print("------- after faiss_huggingface_client.search -----------------")
    print(doc)
docs_hf = faiss_client_hf.search_compression(query,collection_name)
for doc in docs_hf:
    print("------- after faiss_huggingface_client.search_compression -----------------")
    print(doc)

#
###
#
#
#
print("\n\n\n\n")
docs = faiss_client.search_llm(query,collection_name)
print("------- after faiss_client.search_llm -----------------")
print(docs)

docs_openai = faiss_client_openai.search_llm(query,collection_name)
print("------- after faiss_openai_client.search_llm -----------------")
print(docs)

docs_hf = faiss_client_hf.search_llm(query,collection_name)
print("------- after faiss_huggingface_client.search_llm -----------------")
print(docs)
