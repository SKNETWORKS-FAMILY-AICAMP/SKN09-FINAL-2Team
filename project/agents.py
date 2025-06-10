import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# LangChain
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate, SystemMessagePromptTemplate,
    MessagesPlaceholder, HumanMessagePromptTemplate
)
# from langchain_core.agents import create_react_agent
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain.tools import Tool

# 도구들 import
from tools.rag_tool import retrieve_from_qdrant
from tools.rds_tool import MySQLQueryTool
from tools.naver_tool import naver_shop_search

# LLM 초기화
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

system_prompt_text = """
[과정]
- Thought → Action → Action Input → Observation 순으로 최대 4회 반복합니다.
- 각 라벨은 반드시 줄 바꿈으로 분리된 단독 라인으로 작성되어야 하며, 누락되면 실행이 중단됩니다.

[출력 포맷]
- Thought: (도구 선택 이유 및 이전 Observation에 대한 평가)
- Action: (도구 이름만 정확히 입력 (예: rds_tool))
- Action Input: (도구에 보낼 입력, SQL 또는 자연어)
- Observation: (자동 삽입됨, 직접 작성 X)

⚠️ 반드시 Thought 이후에 Action 또는 Final Answer 중 하나가 와야 합니다.
한번 사이클을 돈 뒤 Thought 결과 현재 검색 결과가 충분하다면, Final Answer로 넘어가세요.

최종 추천을 마무리하려면 아래 형식을 반드시 따르세요:
Thought: 조건이 충족되었음을 판단하고 최종 결과를 생성한다.
Final Answer: (상품 추천 내용)

[도구 사용 지침]
- 반드시 [rds_tool, rag_tool, naver_tool] 중 하나만 사용하세요.
- 결과를 직접 생성하지 말고 반드시 Action을 통해 도구를 호출하세요.


[도구 선택 전략]
1. 가격, 카테고리, 브랜드처럼 구조적 조건이 명확하면 `rds_tool`을 우선 사용하세요.
2. 감정, 상황 등 추상적 조건이 많으면 `rag_tool`을 사용하세요.
3. 위 도구 결과가 충분하지 않을 경우 `naver_tool`을 사용하세요.

    [rds_tool]
    rds_tool 사용 규칙:
    - SELECT 쿼리만 사용하세요.
    - 사용할 수 있는 필드: BRAND, NAME, CATEGORY, PRICE
    - BRAND, NAME을 조건으로 사용할 때는 LIKE를 사용하여 검색하세요.
    - 같은 조건에서 다른 상품을 탐색한다면 OFFSET을 변경하여 다음 순서부터 가져온다.

    - CATEGORY는 다음 값만 허용됩니다:
      '유아동', '선물권/교환권', '테마/기념일 선물', '레저/스포츠/자동차', '건강', '식품/음료',
      '디지털/가전', '뷰티', '리빙/인테리어', '반려동물', '패션', '생활', '프리미엄 선물'
    - 잘못된 필드나 CATEGORY 값을 사용하면 쿼리는 무효 처리됩니다.


    rds_tool 쿼리 예시:
    - SELECT NAME, LINK, THUMBNAIL_URL, PRICE FROM PRODUCT WHERE category='뷰티' AND price <= 50000 ORDER BY RAND() LIMIT 5;
    - SELECT NAME, LINK, THUMBNAIL_URL, PRICE FROM PRODUCT WHERE brand='설화수' AND PRICE >= 30000 AND PRICE < 40000 ORDER BY RAND() LIMIT 5;


 [추천 지침]
- 감정(emotion), 스타일(preferred_style), 예산(price_range), 친밀도(closeness) 조건은 반드시 반영하세요.
- 예산(price_range) 범위 예시 : 7만원대 => 70,000원~79,999원
- 예산 초과 금지, 중복 상품 금지.


[Final Answer]
- Final Answer에는 반드시 Observation에 기반하여 4개의 상품만 포함하세요.
- 추천에 앞서 안내하는 문구와 함께 시작하세요:
  - "요청하신 (~)조건에 맞춰 선물을 찾아봤어요!"
  - "(감정)과 (스타일), (예산)에 맞게 아래 상품들을 추천드립니다!"

- 추천 이유: 선물받는 사람과의 관계, 상황, 감정이나 선물의 조건을 충분히 고려하여 해당 상품을 추천하는 이유를 작성하세요.

Final Answer 형식:
1.
- 상품명: ... 
- 가격: ₩xx,xxx
- 이미지: ...
- 링크: ...
- 추천 이유: ... 

[응답 예시]
Thought: 사용자의 감성적 요청과 함께 명확한 가격 조건이 있으므로 먼저 rds_tool로 검색해본다.  
Action: rds_tool  
Action Input: "SELECT NAME, LINK, THUMBNAIL_URL, PRICE FROM PRODUCT WHERE CATEGORY = '테마/기념일 선물' AND PRICE <= 70000 ORDER BY RAND() LIMIT 5;"  
Observation: 관련된 제품이 2개 검색됨.
Thought: 가격은 만족하지만 추천 개수와 감정적인 분위기를 고려해 rag_tool로 보완이 필요하다.
Action: rag_tool  
Action Input: "50대 부모님에게 드릴 감동적인 분위기의 10만원 이하의 결혼기념일 선물"  
Observation: 감성 기반 상품 10개 검색됨.
Thought: 감정과 예산 조건을 모두 만족하는 상품들이 충분히 있으므로, 최종 추천 결과를 생성한다.
Final Answer: 원하시는 선물에 맞는 상품들을 찾아봤어요. 다음은 감동적인 결혼기념일에 어울리는 10만원 이하 선물 추천입니다:
1.  
- 상품명: 감성 캔들 세트  
- 가격: ₩38,000  
- 이미지: https://example.com/candle.jpg  
- 링크: https://giftshop.com/candle  
- 추천 이유: 은은한 향으로 분위기를 더해주는 감성적인 캔들입니다.
"""

# 시스템 프롬프트 정의
system_prompt = SystemMessagePromptTemplate.from_template(system_prompt_text)

# 최종 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("""
{input}

Tools: {tools}
Tool Names: {tool_names}

{agent_scratchpad}
""")
])

# 에이전트 생성 함수
def create_agent():
    # 도구 리스트
    tools = [
        Tool(
            name="rag_tool",
            func=retrieve_from_qdrant,
            description="Qdrant 벡터 데이터베이스에서 유사한 문서 검색"
        ), 
        Tool(
            name="rds_tool",
            func=MySQLQueryTool(
                # host=os.getenv('RDS_HOST', 'localhost'),
                # user=os.getenv('RDS_USER', 'root'),
                # password=os.getenv('RDS_PASSWORD', 1234),
                # database=os.getenv('RDS_DATABASE', 'product_db')
                host=os.getenv('DB_HOST', 'localhost'),
                user=os.getenv('DB_USER', 'root'),
                password=os.getenv('DB_PASSWORD', 1234),
                database=os.getenv('DB_DATABASE', 'senpick_db')
            )._run,
            description="RDS의 MySQL에서 제품 정보 검색"
        ), 
        Tool(
            name="naver_tool",
            func=naver_shop_search,
            description="네이버 쇼핑에서 제품 검색"
        )
    ]
    
    agent = create_react_agent(
        tools=tools,
        llm=llm,
        prompt=prompt
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        max_execution_time=9999,
    )

# llm 외부에서도 사용 가능하도록 export
__all__ = ["create_agent"]