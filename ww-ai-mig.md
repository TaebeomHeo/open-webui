# 주요사항 파악

### RAG

##### Vector DB

- vector db로 chroma를 사용하고 있는데, 한글 처리가 용이한 faiss로 수정 변경해야 함 (📌TODO📌)

##### 파일 upload -> embedding -> vectordb store 과정

- routes/(app)/documents/uploadDocToVectorDB호출(in apis) -> rag_api/doc에서 처리 (함수명: store_doc)
- upload후 처리과정 : UPLOAD_DIR에 올린 후, vectorizing
  - (backend/app/rag) Loader선택(file type에 따라) -> store_data_in_vector_db 호출, 파일명으로 collection_name 만들기, embedding까지 완료
  - 그 이후 createNewDoc (/documents/create호출)으로 객체 생성후-> documents list에 추가

##### 파일 지정후 RAG 기반 query하는 과정

- chatting 창(routes/c/[id]) 에서 submit하면 sendPrompt -> sendPromptOllama까지 흘러들어가서 generateChatCompletion까지 호출하게 됨
- 서버의 ollama/api/chat 호출 -> /ollama/api/chat/{}
- RAGMiddleware에서 rag_messages(backend/rag/utils.py)를 호출하면서 RAG이 실행됨 (단 prompt에 'docs'가 포함되어있다면)

###### rag_message 함수 상세

- prompt에서 받은 doc list를 파악 : 이전 chat에서 언급된 것 까지 모두 포함 -> 중복가능
- docs field는 아래 object의 리스트임
- ```
  {
  'type': 'doc',
  'collection_name': 'b1a6c54ac953d6954767f5d819cf4e72828dc5133eff0ae1fa150d90b93565d',
  'name': '현대차',
  'title': '현대차-그랜져-메뉴얼1.pdf',
  'filename': '현대차-그랜져-메뉴얼1.pdf',
  'content': {},
  'user_id': '2e2f8b67-34d2-48df-bfcc-92a38b10b85c',
  'timestamp': 1715643958,
  'upload_status': True
  }
  ```
- collection_name이 이미 client에서 조회한 후 서버로 올리는 듯
- ✔️ docs 여러개에 대해서 일일이 similarity_query를 수행함(🔔🔊)
- 최상위함수 : query_collection_with_hybrid_search 혹은 query_collection
- query_collection: 📖
  - 각각의 collection에대해서 query_doc 수행후 -> merge_and_sort_query_results로 재조정
  - type:'doc' -> 그 document 하나에 대해서 조회(chroma 용어로는 collection에 대해서 collection.query)
  - type:'collection' -> 문서에 등록된 모든 collection list를 가지고 있음
- merge_and_sort_query_results: 📓💻
  - distances, documents, metadatas tuples을 distance를 기준으로 내림차순
