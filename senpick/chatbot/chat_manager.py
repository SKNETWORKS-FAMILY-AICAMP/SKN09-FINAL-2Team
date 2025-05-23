import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import (
    ChatPromptTemplate, HumanMessagePromptTemplate,
    MessagesPlaceholder, PromptTemplate
)
from langchain_core.messages import HumanMessage, AIMessage

# ✅ .env 로딩
load_dotenv()

# ▶️ 전역 상태 변수
chat_history = []
situation_info = {
    "closeness": "",
    "emotion": "",
    "preferred_style": "",
    "price_range": ""
}
recipient_info = {
    'GENDER': "여성",
    'AGE_GROUP': "30대",
    'RELATION': "연인",
    'ANNIVERSARY': "100일",
}

# ▶️ 시스템 프롬프트 설정
system_message = """
    <시스템 프롬프트>
    당신은 간단한 이벤트용 선물을 추천해주는 챗봇입니다.
    대화 초반에는 사용자에게 필요한 선물의 맥락을 물어보고, 구체적인 선물 제안은 하지 않습니다.

    다음 항목 중 emotion, preferred_style, price_range 이 3가지가 모두 채워졌을 때만 추천을 시작하세요.
    closeness는 선택 항목입니다.

    그 전까지는 반드시 질문만 하며 정보를 유도하세요.
    친근하고 자연스럽게 구어체로 질문해 주세요.

    <안전지침>
    선물과 관련 없는 내용에는 아래와 같이 답변합니다.
    - '죄송합니다. 관련 정보를 확인할 수 없습니다.'
    - '죄송합니다. 선물 추천과 관련 없는 질문에는 답변할 수 없습니다.'
"""
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

situation_info_prompt = PromptTemplate(
    input_variables=["chat_history", "current_info"],
    template="""
    엄격한 정보 추출 가이드라인:
    1. 대화 내용에 명시적으로 언급된 정보만 추출
    2. 추론이나 임의 해석 금지
    3. 언급되지 않은 필드는 빈 문자열로 유지

    대화 내용:
    {chat_history}

    현재 상황 정보:
    {current_info}

    사용자의 응답에서 다음과 같은 정보를 추론하여 추출하세요.
    [추론해야하는 정보]
    "closeness" : 친밀도 (가까움, 어색함, 친해지고 싶음, 애매함 등으로 요약)
    "emotion" : 선물의 동기나 배경이 된 감정 상태 (예: 예의상, 화해, 진심을 전하고 싶음 등).
    "preferred_style" : 희망하는 선물의 스타일, 선물하는 사람의 스타일
    "price_range" : 예산 범위 (3만원~4만원, 7만원 이하, 모름)

    나머지 내용은 최소 1번 이상 질문해야 합니다.
    사용자 답변에 있는 내용만 current_info에서 수정하여 출력합니다.
    사용자 답변이 명확하지 않은 항목은 "없다", "모름" 등의 표현으로 채워도 됩니다.
    코드블럭 없이 JSON 형식으로 정확히 출력하세요.
"""
)

chat_model = None
chat_chain = None
situation_info_chain = None
agent = None

def init_components():
    global chat_model, chat_chain, situation_info_chain, agent
    if chat_model is None:
        chat_model = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        chat_chain = chat_prompt | chat_model
        situation_info_chain = situation_info_prompt | chat_model
    if agent is None:
        from chatbot.agents import create_agent
        agent = create_agent()

def check_situation_info(situation_info):
    required_keys = ["closeness", "emotion", "preferred_style", "price_range"]
    for key in required_keys:
        if situation_info[key] == "" or situation_info[key] in ["없다", "모름", "없음"]:
            return False
    return True

def chat_turn(user_input):
    global chat_history, situation_info, recipient_info
    init_components()

    response = chat_chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.content))

    situation_info_response = situation_info_chain.invoke({
        "chat_history": chat_history,
        "current_info": json.dumps(situation_info)
    })
    situation_info = json.loads(situation_info_response.content)

    agent_reply = None
    if check_situation_info(situation_info):
        agent_output = agent.invoke({
            "input": f"{recipient_info['AGE_GROUP']} {recipient_info['GENDER']} {recipient_info['RELATION']}에게 {recipient_info['ANNIVERSARY']} 선물 추천 쿼리: {situation_info}",
            "chat_history": chat_history
        })
        agent_reply = agent_output['output']

    return response.content, agent_reply
