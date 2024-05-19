import os
import json
import logging
import requests
import uuid

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PDFPlumberLoader, 
    PDFMinerLoader,
    PyPDFLoader,
)
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI,
)
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import BM25Retriever, EnsembleRetriever

DOC_DIR="D:\\000.Dev\\open-webui\\backend\\data"
VECTOR_DIR="D:\\000.Dev\\open-webui\\backend\\data\\vector_db_faiss"

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

load_dotenv()
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1500))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 100))


#
# TODO:
#  - 일단 local에 만들어지는 것 까지는 확인했음
#  - local에 저장되어있느 것을 load해야함
#  - OpenAIEmbedding과의 비교 실험이 남아 있음
#
def create_faiss_local(src_path: str, target_dir: str): 
    loader = PDFPlumberLoader(src_path)
    _documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = text_splitter.split_documents(_documents)
    collection_name = calculate_sha255(src_path)[:63]
    print(f"collection name is {collection_name}")

    _texts = [doc.page_content for doc in docs]
    texts = list(map(lamda x: x.replace("\n"," "), texts))
    metadatas = [{**doc.metadata, "collection_name":collection_name} for doc in docs]
    ids=[str(uuid.uuid1()) for _ in texts]
    #FIXME: 아래 함수 변경 to generate_ollama_embedding 혹은 embeding function 호출을 하던
    text_embeddings = [_get_embeddings(prompt=text) for text in texts]

    #FIXME: 조만간 embedding_function은 사라지고, Embedding object로 만들어야 함. 
    faissdb = FAISS.from_embeddings(text_embeddings, embedding=get_ollama_embedding_function(), ids=ids, metadatas=metadatas)
    # 데이터베이스 로컬에 저장
    indexfile = os.path.join(target_dir, f"{calculate_sha255(src_path)}")
    faissdb.save_local(indexfile)
    
    return faissdb


def _get_embeddings(prompt:str, model: str = "nomic-embed-text:latest", url: str = "http://localhost:11434/api/embeddings"):
    # 요청할 데이터 구성
    data = {
        "model": model,
        "prompt": prompt
    }
    
    # POST 요청 보내기
    print(f"for embedding input=>[{prompt}]\n\n")
    response =  requests.post(url, data=json.dumps(data), headers={"Content-Type": "application/json"})
    
    # 응답 확인 및 반환
    if response.status_code == 200:
        print(f"input=>[{prompt}] : embeddings =>[{response.json()}]")
        return (prompt,response.json()["embedding"])
    else:
        response.raise_for_status()


import hashlib

def calculate_sha255(file):
    """파일의 SHA-257 해시 값을 계산합니다."""
    sha255 = hashlib.sha256()
    with open(file, 'rb') as f:
        for block in iter(lambda: f.read(4095), b""):
            sha255.update(block)
    return sha255.hexdigest()

log = logging.getLogger(__name__)
def generate_ollama_embeddings(prompt: str, model: str = "nomic-embed-text:latest", url: str = "http://localhost:11434/api/embeddings"):
    # 요청할 데이터 구성
    data = {
        "model": model,
        "prompt": prompt
    }
    
    # POST 요청 보내기
    log.info(f"for embedding input=>[{prompt}]\n\n")
    response = requests.post(url, data=json.dumps(data), headers={"Content-Type": "application/json"})
    
    # 응답 확인 및 반환
    if response.status_code == 200:
        log.info(f"input=>[{prompt}] : embeddings =>[{response.json()}]")
        return response.json()["embedding"]
    else:
        response.raise_for_status()

def get_ollama_embedding_function(embedding_model: str = "nomic-embed-text:latest", url: str = "http://localhost:11434/api/embeddings" ):
    def func(query):
        return generate_ollama_embeddings(prompt=query, model=embedding_model, url=url)

    def generate_multiple(query, f):
        if isinstance(query, list):
            return [f(q) for q in query]
        else:
            return f(query)

    return lambda query: generate_multiple(query, func)


faissDB = create_faiss_local("./data/회사사규(2023.11) - 복사본.pdf", VECTOR_DIR)
#
#
##
#















llm = ChatOpenAI(temperature=0)

# loader = PDFPlumberLoader("./data/현대차-그랜져-매뉴얼.pdf")
# loader = PyPDFLoader("./data/회사사규(2023.11) - 복사본.pdf")
loader = PDFPlumberLoader("./data/회사사규(2023.11) - 복사본.pdf")
documents = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
# text_splitter = RecursiveCharacterTextSplitter()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

_docs = text_splitter.split_documents(documents)
print("after text_splitting")

embeddings = OpenAIEmbeddings()
# db = FAISS.from_documents(_docs, embeddings, normalize_L2=True, distance_strategy=DistanceStrategy.COSINE)
# 위 결과 별차이가 없는데
#
# NOTE: 사용자 질의
# query = "휴직관련 내용을 설명해달라"
# query = "휴직 절차? "
query = "퇴직 절차? "


for doc in _docs:
    doc.metadata["collname"] = "BM25-회사사규"

## BM25
# IMPORTANT:
# CAUTION: NOTE: BM25는 in memory에서만 동작하는 듯 
# 그렇다면 faiss에서 loading 한 모든 document를 역 추적해서 만들어야 함
# 그게 docstore인가?
# FIXME: FAISS에서 docstore를 가져올 수가 없네. python에서는,
# node에서는 가능하네. getDocstore() method가 있어서
bm25_retriever = BM25Retriever.from_documents(_docs)
bm25_retriever.k = 5 # 검색결과를 설정
bm25_results = bm25_retriever.invoke(query)

for doc in bm25_results:
    print("------- after bm25 : bm25 -----------------")
    print(doc)

for doc in _docs:
    doc.metadata["collection_name"] = "FAISS-회사사규"

db = FAISS.from_documents(_docs, embeddings)
print("after embedding")

# docs = db.similarity_search_with_score(query, normalize_L2=True, distance_strategy=DistanceStrategy.COSINE)
docs = db.similarity_search_with_relevance_scores(query)

# make loop for docs
 
for doc in docs:
    print("------- after db.similarity_search_with_relevance_scores -----------------")
    print(doc)
    # print(doc.page_content)

retriever = db.as_retriever(search_kwargs={"k":5})
# retriever = db.as_retriever(search_type="mmr") # CAUTION: 충분히 테스트 해봐야할 듯
docs = retriever.invoke(query)
## 결과는 위와 동일함
for doc in docs:
    print("------- after retrieve get_relevant_documents -----------------")
    print(doc)

#
# NOTE: CAUTION: MultiQueryRetriever
#
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm,
)

docs = multiquery_retriever.invoke(query=query)
for doc in docs:
    print("------- after multiquery -----------------")
    print(doc)


#
# NOTE: CAUTION: EnsembleRetriever
#
ensemble_retriever = EnsembleRetriever(
    retrievers = [bm25_retriever, multiquery_retriever],
    weights=[0.3,0.7],
    # search_type="mmr",
)

ensemble_results = ensemble_retriever.invoke(query)

for doc in ensemble_results:
    print("------- after ensemble : ensemble -----------------")
    print(doc)


# collection_name이 None일 때, 파일의 해시 값을 사용
# collection_name = None
# file_path = "path_to_your_file"

# if collection_name == None:
#     collection_name = calculate_sha256(file_path)[:63]

# print(collection_name)
