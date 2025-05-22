import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import (
    ChatPromptTemplate, HumanMessagePromptTemplate,
    MessagesPlaceholder, PromptTemplate
)
from agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

# 채팅 모델 선언
chat_model = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-4o-mini",
)

agent = create_agent()

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
    - '죄송합니다. 선물 추천과 관련 없는 질문에는 답변할 수 없습니다.
"""
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}")
])
# 상황 정보 프롬프트 설정
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
""")
    # ▶️ GPT 모델 초기화

chat_chain = chat_prompt | chat_model

situation_info_chain = situation_info_prompt | chat_model

def check_situation_info(situation_info):
    """
    상황 정보가 모두 채워졌는지 확인하는 함수
    :param situation_info: dict, 상황 정보
    :return: bool, 모든 정보가 채워졌는지 여부
    """
    required_keys = ["closeness", "emotion", "preferred_style", "price_range"]
    for key in required_keys:
        if situation_info[key] == "" or situation_info[key] in ["없다", "모름", "없음"]:
            return False
    return True


def chat():
    # 🎯 사용자 정보
    recipient_info = {
        'GENDER': "여성",
        'AGE_GROUP': "30대",
        'RELATION': "연인",
        'ANNIVERSARY': "100일",
    }
    
    # ▶️ 상황 정보 초기 상태
    situation_info = {
        "closeness": "",
        "emotion": "",
        "preferred_style": "",
        "price_range": ""
    }

    # 🎯 대화 이력 초기화
    chat_history = []

    # 🎯 사용자 메시지 생성
    user_message = f"""
        다음 정보를 바탕으로 기념일 선물을 추천해줘.
        성별: {recipient_info['GENDER']}
        연령대: {recipient_info['AGE_GROUP']}
        관계: {recipient_info['RELATION']}
        기념일 종류: {recipient_info['ANNIVERSARY']}
    """

    # 🧠 LLM 체인 호출 (기본 응답)
    response = chat_chain.invoke({
        "input": user_message,
        "chat_history": chat_history
    })
    chat_history.append(HumanMessage(content=user_message))
    chat_history.append(AIMessage(content=response.content))
    print("응답:", response.content)
    while True:
        # 🎯 대화 이력 업데이트
        input_message = input("사용자 입력 (종료: 'exit'): ")
        if input_message.lower() == '종료':
            print("챗봇 종료.")
            break
        res = chat_chain.invoke({
            "input": input_message,
            "chat_history": chat_history
        })
        chat_history.append(HumanMessage(content=input_message))
        print(res.content)
        
        # 🧠 LLM 체인 호출 (상황 정보 업데이트)
        situation_info_response = situation_info_chain.invoke({
            "chat_history": chat_history,
            "current_info": json.dumps(situation_info)
        })
        print(situation_info_response)
        # 🎯 상황 정보 업데이트
        situation_info = json.loads(situation_info_response.content)
        print(situation_info)

        # 🎯 대화 이력 업데이트
        chat_history.append(AIMessage(content=res.content))
        
        # ✅ 상황 정보가 모두 채워졌는지 확인
        if check_situation_info(situation_info):
            print("🎯 상황 정보가 모두 채워졌습니다. 에이전트에게 쿼리를 보냅니다...")
            res = agent.invoke({
                "input": f"기념일 선물 추천을 위한 쿼리: {situation_info}",
                "chat_history": chat_history
            })
            print("에이전트 응답:", res['output'])

# ▶️ 진입점
if __name__ == "__main__":
    chat()