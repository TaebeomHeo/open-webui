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
