from langchain.tools import Tool                  # Tool 클래스를 사용하기 위함
from langchain_qdrant import QdrantVectorStore         # Qdrant 벡터스토어
from langchain_huggingface import HuggingFaceEmbeddings  # 임베딩 모델
from qdrant_client import QdrantClient
import time 

# Docker에서 실행 중인 Qdrant에 연결
# 프로젝트 루트 디렉토리에서 Qdrant 실행 (구글 드라이브의 qdrant_storage 다운받아 루트에 압축 해제 (SKN09-FINAL-2TEAM/qdrant_storage))
# docker run -d -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage --name qdrant qdrant/qdrant (맥 환경)
# docker run -d -p 6333:6333 -p 6334:6334 -v ${PWD}/qdrant_storage:/qdrant/storage --name qdrant qdrant/qdrant (윈도우 파워쉘 환경 가능, cmd에서는 사용 불가)
qdrant = QdrantClient(url="http://localhost:6333", prefer_grpc=False)

embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

vectorstore = QdrantVectorStore(
    client=qdrant,
    collection_name="senpick",
    embedding=embedding,
)

def retrieve_from_qdrant(query: str) -> str:
    try:
        # 검색 결과 수 10개로 확장, 더 다양한 결과 제공
        start = time.time()
        results = vectorstore.similarity_search_with_score(query, k=10)
        elapsed = time.time() - start  # 시간 측정
        print(f"[RAG] 🔍 검색 소요 시간: {elapsed:.2f}초")
        
        # 결과가 없으면 빈 문자열 반환
        if not results:
            return ""
        
        # 결과 포맷팅: 각 문서에 대한 상세 정보 제공
        formatted_results = []
        for i, doc in enumerate(results, 1):
            content = {k: doc[0].metadata[k] for k in ['title', 'price', 'product_url', 'thumbnail_url']}
            formatted_result = f"\n[결과 {i}]\n내용: {content}\n유사도: {doc[1]}\n"
            formatted_results.append(formatted_result)
        
        return "\n\n".join(formatted_results)
    except Exception as e:
        print(f"RAG 검색 중 오류 발생: {e}")
        return ""

qdrant_rag_tool = Tool(
    name="QdrantSearch",
    func=retrieve_from_qdrant,
    description="Qdrant에서 유사 문서를 검색합니다. 문서 기반 질문에 사용하세요."
)

__all__ = ["qdrant_rag_tool"]