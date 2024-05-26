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
from langchain.storage import InMemoryStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.neo4j_vector import SearchType
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever, RePhraseQueryRetriever, ContextualCompressionRetriever, ParentDocumentRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader, PyPDFLoader
from dotenv import load_dotenv

# 로그 레벨 설정 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

# 로거 객체 생성
logger = logging.getLogger(__name__)
load_dotenv()

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1500))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 100))
PARENT_CHUNK_SIZE = int(os.environ.get("PARENT_CHUNK_SIZE", 2000))
PARENT_CHUNK_OVERLAP = int(os.environ.get("PARENT_CHUNK_OVERLAP", 100))
CHILD_CHUNK_SIZE = int(os.environ.get("CHILD_CHUNK_SIZE", 500))
CHILD_CHUNK_OVERLAP = int(os.environ.get("CHILD_CHUNK_OVERLAP", 20))
os_is_mac = (os.getenv("OS_SUPPORT") == "MAC") 

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    #
    # logging.info(f"{file_path}의 sha256 값 : {sha256_hash.hexdigest()}")
    #
    return sha256_hash.hexdigest()

class FAISS_CLIENT:
    def __init__(self, vector_db_dir, ollama_llm_model, ollama_embedding_model, ollama_url, type="ollama"):
        self.vector_db_dir = vector_db_dir
        self.ollama = {}
        if type == "ollama" :
            self.ollama['llm'] = ChatOllama(base_url=ollama_url, model=ollama_llm_model)
            self.ollama['embedding'] = OllamaEmbeddings(model=ollama_embedding_model, base_url=ollama_url)
        elif type == "openai":
            self.ollama['llm'] = ChatOpenAI(temperature=0) 
            self.ollama['embedding'] = OpenAIEmbeddings()
        elif type == "huggingface":
            #FIXME: not working for M1
            self.ollama['llm'] = ChatOllama(base_url=ollama_url, model=ollama_llm_model)
            self.ollama['embedding'] = HuggingFaceEmbeddings(
                model_name = "sentence-transformers/all-MiniLM-L6-v2",
                # model_kwargs={"device":"cpu"},
                model_kwargs={"device":"mps"}, # for M1
                encode_kwargs={"normalize_embeddings":False}
            )
            
        self.collections = {}
        # self.load_collections()


    def load_collection_by_ext(self, directory, ext):
        """Load collections for all files with the specified extension in the given directory."""
        self.doc_dir = directory
        for file in os.listdir(directory):
            if file.endswith(ext):
                file_path = os.path.join(directory, file)
                self.create_collection(file_path)

    def create_collection(self, src_pdf_path):
        """Create a FAISS collection from a PDF file."""
        index_name = calculate_sha256(src_pdf_path)[:63]
        #
        logging.info(f"create new collection [{index_name}] from [{src_pdf_path}]")
        #
        if index_name in self.collections:
            #
            logging.info(f"Collection with name {index_name} already exists.")
            #
            # raise ValueError(f"Collection with name {index_name} already exists.")
            return index_name

        faiss_db, retriever = self.create_faiss_db_and_retriever(src_pdf_path, index_name)
        rag_chain = self.create_rag_chain(retriever) if retriever else None
        compression_retriever = ContextualCompressionRetriever(
                    # base_compressor= LLMChainExtractor.from_llm(self.ollama["llm"]),
                    base_compressor= LLMChainFilter.from_llm(self.ollama["llm"]),
                    base_retriever = retriever,
        )
        rag_chain_compression = self.create_rag_chain(compression_retriever) if compression_retriever else None

        # for ParentDocumentRetriever
        parent_doc_retriever = self.create_faiss_db_and_parent_doc_retriever(src_pdf_path)
        rag_chain_parent_doc = self.create_rag_chain(parent_doc_retriever) 

        self.collections[index_name] = {
            "faiss_db": faiss_db,
            "retriever": retriever,
            "compression_retriever": compression_retriever, 
            "rag_chain": rag_chain,
            "rag_chain_compression": rag_chain_compression,
            "parent_doc_retriever": parent_doc_retriever,
            "rag_chain_parent_doc": rag_chain_parent_doc,
        }
        
        return index_name

    def delete_collection(self, collection_name):
        """Delete a FAISS collection."""
        #
        logging.info(f"deleting... collection [{collection_name}]")
        #
        if collection_name in self.collections:
            del self.collections[collection_name]
            os.remove(os.path.join(self.vector_db_dir, f"{collection_name}.faiss"))
        else:
            raise ValueError(f"Collection {collection_name} does not exist.")

    def search(self, query, collection_name):
        """Search the given query in the specified collection."""
        #
        logging.info(f"searching [{query}] in collection [{collection_name}]")
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

    def search_compression(self, query, collection_name):
        """Search the given query in the specified collection."""
        #
        logging.info(f"searching using compression [{query}] in collection [{collection_name}]")
        #
        if collection_name == "all":
            results = []
            for name, data in self.collections.items():
                retriever = data["compression_retriever"]
                result = retriever.invoke(query)
                results.extend(result)
            return results
            # return ContextCompress(results).compress()
        else:
            if collection_name not in self.collections:
                raise ValueError(f"Collection {collection_name} does not exist.")
            retriever = self.collections[collection_name]["compression_retriever"]
            return retriever.invoke(query)

    def search_parent_doc(self, query, collection_name):
        """Search the given query in the specified collection."""
        #
        logging.info(f"searching using parentDocRetriever [{query}] in collection [{collection_name}]")
        #
        if collection_name == "all":
            results = []
            for name, data in self.collections.items():
                retriever = data["parent_doc_retriever"]
                result = retriever.invoke(query)
                results.extend(result)
            return results
            # return ContextCompress(results).compress()
        else:
            if collection_name not in self.collections:
                raise ValueError(f"Collection {collection_name} does not exist.")
            retriever = self.collections[collection_name]["parent_doc_retriever"]
            return retriever.invoke(query)
        
    def search_llm(self, query, collection_name):
        """Search the given query in the specified collection."""
        #
        logging.info(f"searching using rag_chain [{query}] in collection [{collection_name}]")
        #
        if collection_name == "all":
            results = []
            for name, data in self.collections.items():
                rag_chain = data["rag_chain"]
                result = rag_chain.invoke(query)
                results.extend(result)
            return results
            # return ContextCompress(results).compress()
        else:
            if collection_name not in self.collections:
                raise ValueError(f"Collection {collection_name} does not exist.")
            rag_chain = self.collections[collection_name]["rag_chain"]
            return rag_chain.invoke({'input':query})

    def search_llm_compression(self, query, collection_name):
        """Search the given query in the specified collection."""
        #
        logging.info(f"searching using rag_chain_compression [{query}] in collection [{collection_name}]")
        #
        if collection_name == "all":
            results = []
            for name, data in self.collections.items():
                rag_chain = data["rag_chain_compression"]
                result = rag_chain.invoke(query)
                results.extend(result)
            return results
            # return ContextCompress(results).compress()
        else:
            if collection_name not in self.collections:
                raise ValueError(f"Collection {collection_name} does not exist.")
            rag_chain = self.collections[collection_name]["rag_chain_compression"]
            return rag_chain.invoke({'input':query})

    def search_llm_parent_doc(self, query, collection_name):
        """Search the given query in the specified collection."""
        #
        logging.info(f"searching using rag_chain_parent_doc [{query}] in collection [{collection_name}]")
        #
        if collection_name == "all":
            results = []
            for name, data in self.collections.items():
                rag_chain = data["rag_chain_parent_doc"]
                result = rag_chain.invoke(query)
                results.extend(result)
            return results
            # return ContextCompress(results).compress()
        else:
            if collection_name not in self.collections:
                raise ValueError(f"Collection {collection_name} does not exist.")
            rag_chain = self.collections[collection_name]["rag_chain_parent_doc"]
            return rag_chain.invoke({'input':query})
        
    def get_collections(self):
        """Return a list of all collection names."""
        return list(self.collections.keys())
    
    def get_retriever(self, faiss_db, type='vectordb'):
        """Return the appropriate retriever based on the specified type."""
        if type == 'vectordb':
            return faiss_db.as_retriever(
                # search_type="similarity",
                search_type="mmr",
                search_kwargs={
                    "k": 10,
                    # "lambda_mult": 0.25, # 매우 유사한 것이 많을떄, in mmr
                    "fetch_k": 50, # 'mmr인 경우, 후보군 확대
                }
            )
        elif type == 'multiquery':
            return MultiQueryRetriever.from_llm(
                retriever=faiss_db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 10}
                ),
                llm=self.ollama['llm'],
                include_original = True, # for MultiQuery only
            )
        elif type == 'rephrase':
            return RePhraseQueryRetriever.from_llm(
                retriever=faiss_db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 10}
                ),
                llm=self.ollama['llm']
            )
        elif type == 'fullchain': ## 아마 필요없을 듯
            retriever = RePhraseQueryRetriever.from_llm(
                retriever=faiss_db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 10}
                ),
                llm=self.ollama['llm']
            )
            return self.create_rag_chain(retriever)
        else:
            raise ValueError(f"Unknown retriever type: {type}")

    def create_rag_chain(self, retriever):
        """Create a RAG chain for the given retriever."""
        prompt = ChatPromptTemplate.from_template("You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.  Question: {input} Context: {context} Answer:")
        llm = self.ollama['llm']
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)
        return chain

        # NOTE: CAUTION: 위와 아래의 결과가 동일함. LCEL을 함수로 만들어 놓은 것임
        # output_parser = StrOutputParser()

        # def format_docs(docs):
        #     return "\n\n".join(doc.page_content for doc in docs)
        # # format_docs = lambda x: x  # placeholder for document formatting
        # return (
        #     # FIXME: TODO: DONE: ContextCompressionRetriever와 format_docs는 piping오류 발생 (tuple vs function)
        #     # syntax error였음
        #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
        #     | prompt 
        #     | self.ollama['llm']
        #     | output_parser 
        # )

    def create_faiss_db_and_retriever(self, src_pdf_path, index_name):
        """Create a FAISS database and retriever from a PDF file."""

        # loader = PDFPlumberLoader(src_pdf_path)
        loader = PyPDFLoader(src_pdf_path)
        _documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        docs = text_splitter.split_documents(_documents)

        embeddings = self.ollama['embedding']
        texts = [doc.page_content.replace("\n", " ") for doc in docs]
        #
        # FIXME: TODO: 아마도 아래 metadata 셋팅하는게 필요없을 듯. 그냥 원래 doc.metadata만 있어도 될 듯
        #
        metadatas = [{"collection_name": index_name, **doc.metadata} for doc in docs]
        ids = [str(uuid.uuid1()) for _ in texts]
        text_embeddings = embeddings.embed_documents(texts)
        text_embedding_pairs = zip(texts, text_embeddings)

        faiss_db = FAISS.from_embeddings(
            text_embedding_pairs, 
            embedding=embeddings, 
            ids=ids, metadatas=metadatas)
        # faiss_db.save_local(self.vector_db_dir, metadatas[0]['collection_name'])

        retriever = self.get_retriever(faiss_db, type='vectordb')
        return faiss_db, retriever

    def create_faiss_db_and_parent_doc_retriever(self, src_pdf_path):
        """Create ParentDocumntRetriever from a PDF file."""

        # loader = PDFPlumberLoader(src_pdf_path)
        loader = PyPDFLoader(src_pdf_path)
        _documents = loader.load()
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PARENT_CHUNK_SIZE, 
            chunk_overlap=PARENT_CHUNK_OVERLAP
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE, 
            chunk_overlap=CHILD_CHUNK_OVERLAP
        )
        store = InMemoryStore()
        embeddings = self.ollama['embedding']
        #
        # FIXME: Chroma처럼 바로 쉽게 만들 수는 없네. 우회적인 방법을..
        faiss_db = FAISS.from_texts(['__a','__b'], embeddings)
        retriever = ParentDocumentRetriever(
            vectorstore = faiss_db,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            # search_type=SearchType.mmr, # default 'SearchType.similarity' FIXME: 오류발생
            # FIXME: 아래를 하니 오히려 더 안좋아짐
            # search_kwargs={
            #     "k": 10,
            #     # "lambda_mult": 0.25, # 매우 유사한 것이 많을떄, in mmr
            #     # "fetch_k": 50, # 'mmr인 경우, 후보군 확대
            # }
        )
        retriever.add_documents(_documents)

        return retriever

    ##
    #
    # CAUTION: NOTE:
    # FIXME: 아래 load_collection은 제거 예정임. be depricated
    # 왜냐하면, parentDocumentRetriever, ensembleRetriever등을 사용하려면
    # on the fly로 embedding 해야 함. 물론 추후에 이런 것이 지원되면 다르겠지만.
    #
    def load_collections(self):
        """Load all FAISS indexes from the vector_db_dir."""

        for file in os.listdir(self.vector_db_dir):
            if file.endswith(".faiss"):
                index_name = file.replace(".faiss", "")
                faiss_db = FAISS.load_local(
                    self.vector_db_dir, 
                    self.ollama['embedding'],
                    index_name,
                    allow_dangerous_deserialization=True,
                    # kwargs={"distance_strategy":DistanceStrategy.COSINE,}
                    distance_strategy=DistanceStrategy.COSINE
                )
                retriever = self.get_retriever(faiss_db, type='vectordb')
                rag_chain = self.create_rag_chain(retriever) if retriever else None
                template_string = (
                    "Given the following question and context, return YES if the context is relevant to the question and NO if it isn't.\n\n"
                    "Answer with YES or NO only. No additional text.\n\n"
                    "> Question: {question}\n"
                    "> Context:\n"
                    ">>>\n"
                    "{context}\n"
                    ">>>\n"
                    "> Relevant (YES / NO):"
                )
                prompt = PromptTemplate.from_template(template=template_string)
                compression_retriever = ContextualCompressionRetriever(
                    # base_compressor= LLMChainExtractor.from_llm(self.ollama["llm"]),
                    base_compressor= LLMChainFilter.from_llm(self.ollama["llm"], prompt),
                    base_retriever = retriever,
                )
                rag_chain_compression = self.create_rag_chain(compression_retriever) if compression_retriever else None
                self.collections[index_name] = {
                    "faiss_db": faiss_db,
                    "retriever": retriever,
                    "compression_retriever": compression_retriever,
                    "rag_chain" : rag_chain,
                    # IMPORTANT:
                    # NOTE: MEMORY: retriever기반의 rag_chain은 거의 답변이 동일함 (local vs openai)
                    # CAUTION: 단지 시간소요가 많이됨
                    "rag_chain_compression":rag_chain_compression
                }
                #
                logging.info(f"adding collection[{index_name}]")
                #