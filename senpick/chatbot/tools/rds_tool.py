#rdstool
import traceback                      # 예외 트레이스백 출력
import mysql.connector                # MySQL DB 연결
from typing import Type              # args_schema 타입 힌트용
from pydantic import BaseModel        # Tool 입력 타입 정의용
from langchain.tools import BaseTool  # LangChain 사용자 정의 Tool 기반 클래스
from pathlib import Path
from dotenv import load_dotenv
import time

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class MySQLQueryInput(BaseModel):
    query: str

class MySQLQueryTool(BaseTool):
    name: str = "mysql_query_tool"
    description: str = """RDS의 MySQL 데이터베이스에서 상품 및 관련 정보를 검색합니다. 반드시 SELECT 쿼리만 사용해야 하며, 상품 이름, 가격대, 브랜드, 카테고리 등 구체적인 정보를 검색할 때 유용합니다. 복잡한 조건 필터링에 강점이 있습니다. 
        The `PRODUCT` table has the following structure:
        - BRAND: Brand name
        - NAME: Product name
        - CATEGORY: Main category
        - SUB_CATEGORY: Subcategory
        - OPTIONS: Options such as size, color, etc.
        - PRICE: Price (integer, in KRW)

        Supported main categories (`CATEGORY` values):
        '유아동', '선물권/교환권', '테마/기념일 선물', '레저/스포츠/자동차',
        '건강', '식품/음료', '디지털/가전', '뷰티',
        '리빙/인테리어', '반려동물', '패션', '생활', '프리미엄 선물'
    """
    args_schema: Type[BaseModel] = MySQLQueryInput

    host: str
    user: str
    password: str
    database: str

    def _run(self, query: str):
        if not query.strip().lower().startswith("select"):
            return "오직 SELECT 쿼리만 허용됩니다."

        conn = None
        cursor = None
        try:
            start = time.time()  # 시작
            conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()            
            elapsed = time.time() - start  # 끝
            print(f"[RDS] 🔍 DB 쿼리 소요 시간: {elapsed:.2f}초")

            column_names = [desc[0] for desc in cursor.description]

            result = [dict(zip(column_names, row)) for row in rows]
            return result or "결과 없음"
        except Exception as e:
            return f"오류 발생: {e}\n{traceback.format_exc()}"
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def _arun(self, query: str):
        raise NotImplementedError("비동기 버전은 지원하지 않습니다.")
    

rds_tool = MySQLQueryTool(
    name="rds_query_tool",
    host="localhost",
    user="root",
    password="root",
    database="senpick_db"
)
__all__ = ["rds_tool"]