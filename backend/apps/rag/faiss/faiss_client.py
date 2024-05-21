#
# local llm, embedding, FAISS를 사용하여 CHROMA-client와 유사하게 만듦
#
# 클래스명 : CHROMA_CLIENT
# 생성시 parameter : VECTOR_DB_DIR, Ollama LLM model, Ollama Embedding model, Ollama url
# methods
#  1. create_collection(src_pdf_path) 
#       src_pdf_path의 PDF를 로딩한 후, 이를 local FAISS로 저장함
#       이때, index name은 src_pdf_path기반 sha256 string으로 함
#       또한, 이미 생성된 index가 있다면 error를 raising함
#  2. delete_collection(collection_name)
#  3. scan_load() : VECTOR_DB_DIR에서 *.faiss 를 읽어서 faiss db dictionary로 생성
#       {index_name: faiss_db}
#       또한, search를 대비해서 collection별로 MultiQueryRetrieval을 만들어 놓아야 함
#  4. search(query, collection_name) : query를 받아들여서, 해당 collection에서 조회함
#     단, 조회는 collection_name에 해당하는 MultiQueryRetrieval로 조회해서 return함
#     만약, collection_name == 'all'이면, 모든 collection에 대해서 query하되
#     각각의 결과들을 모아서 ContextCompress를 해서 return해야 함

import os
import hashlib
import logging
import uuid
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import BM25Retriever, EnsembleRetriever, RePhraseQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader, PyPDFLoader
from dotenv import load_dotenv

# 로그 레벨 설정 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

# 로거 객체 생성
logger = logging.getLogger(__name__)
load_dotenv()

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1500))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 100))

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    #
    logging.debug(f"{file_path}의 sha256 값 : {sha256_hash.hexdigest()}")
    #
    return sha256_hash.hexdigest()

class FAISS_CLIENT:
    def __init__(self, vector_db_dir, ollama_llm_model, ollama_embedding_model, ollama_url):
        self.vector_db_dir = vector_db_dir
        self.ollama = {}
        self.ollama['llm'] = ChatOllama(base_url=ollama_url, model=ollama_llm_model)
        self.ollama['embedding'] = OllamaEmbeddings(model=ollama_embedding_model, base_url=ollama_url)
        self.collections = {}
        self.load_collections()

    def load_collections(self):
        """Load all FAISS indexes from the vector_db_dir."""
        prompt = ChatPromptTemplate.from_template("You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.  Question: {question} Context: {context} Answer:")
        output_parser = StrOutputParser()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        for file in os.listdir(self.vector_db_dir):
            if file.endswith(".faiss"):
                index_name = file.replace(".faiss", "")
                faiss_db = FAISS.load_local(self.vector_db_dir, self.ollama['embedding'],index_name,allow_dangerous_deserialization=True)
                retriever=faiss_db.as_retriever(
                    # search_type="mmr", 
                    search_type="similarity", 
                    search_kwargs={
                        "k":10, 
                        # "lambda_mult":0.25, # 매우 유사한 문장이 반복되는 경우 in MMR
                        # "fetch_k":50 # 'mmr'인 경우만 의미가 있음
                    }
                ) 
                # retriever = MultiQueryRetriever.from_llm( # 품질이 별로인데
                # retriever = RePhraseQueryRetriever.from_llm( # 얘도 황당하게 하네
                #     retriever=faiss_db.as_retriever(
                #         # search_type="mmr", 
                #         search_type="similarity", 
                #         search_kwargs={
                #             "k":10, 
                #             # "lambda_mult":0.25, # 매우 유사한 문장이 반복되는 경우 in MMR
                #             "fetch_k":50 # 'mmr'인 경우만 의미가 있음
                #         }
                #     ), 
                #     llm=self.ollama['llm'],
                #     # include_original=True, # for MultiQueryRetriever only
                # )
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | self.ollama['llm']
                    | StrOutputParser()
                )
                self.collections[index_name] = {
                    "faiss_db": faiss_db,
                    "retriever": retriever,
                    "rag_chain" : rag_chain,
                }
                #
                logging.debug(f"adding collection[{index_name}]")
                #

    def create_collection(self, src_pdf_path):
        """Create a FAISS collection from a PDF file."""
        index_name = calculate_sha256(src_pdf_path)[:63]
        #
        logging.debug(f"create new collection [{index_name}] from [{src_pdf_path}]")
        #
        if index_name in self.collections:
            #
            logging.info(f"Collection with name {index_name} already exists.")
            #
            # raise ValueError(f"Collection with name {index_name} already exists.")
            return index_name

        loader = PDFPlumberLoader(src_pdf_path)
        _documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        docs = text_splitter.split_documents(_documents)

        embeddings = self.ollama['embedding']
        texts = [doc.page_content.replace("\n", " ") for doc in docs]
        metadatas = [{"collection_name": index_name, **doc.metadata} for doc in docs]
        ids = [str(uuid.uuid1()) for _ in texts]
        text_embeddings = embeddings.embed_documents(texts)
        text_embedding_pairs = zip(texts, text_embeddings)

        faiss_db = FAISS.from_embeddings(text_embedding_pairs, embedding=embeddings, ids=ids, metadatas=metadatas)
        faiss_db.save_local(self.vector_db_dir, index_name)

        retriever = MultiQueryRetriever.from_llm(
            retriever=faiss_db.as_retriever(
                search_type="mmr", 
                search_kwargs={
                    "k":10, 
                    "lambda_mult":0.25, # 매우 유사한 문장이 반복되는 경우
                    "fetch_k":50
                    }
                ), 
            llm=self.ollama['llm']
        )
        self.collections[index_name] = {
            "faiss_db": faiss_db,
            "retriever": retriever,
        }

        return index_name

    def delete_collection(self, collection_name):
        """Delete a FAISS collection."""
        #
        logging.debug(f"deleting... collection [{collection_name}]")
        #
        if collection_name in self.collections:
            del self.collections[collection_name]
            os.remove(os.path.join(self.vector_db_dir, f"{collection_name}.faiss"))
        else:
            raise ValueError(f"Collection {collection_name} does not exist.")

    def search(self, query, collection_name):
        """Search the given query in the specified collection."""
        #
        logging.debug(f"searching [{query}] in collection [{collection_name}]")
        #
        if collection_name == "all":
            results = []
            for name, data in self.collections.items():
                retriever = data["retriever"]
                result = retriever.invoke(query)
                results.extend(result)
            return results
            # return ContextCompress(results).compress()
        else:
            if collection_name not in self.collections:
                raise ValueError(f"Collection {collection_name} does not exist.")
            retriever = self.collections[collection_name]["retriever"]
            return retriever.invoke(query)

    def get_collections(self):
        """Return a list of all collection names."""
        return list(self.collections.keys())