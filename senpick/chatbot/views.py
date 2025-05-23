from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .chat_manager import chat_turn
from django.shortcuts import render

def chatbot_page(request):
    return render(request, "chatbot.html")

@csrf_exempt
def chat_respond(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_msg = data.get("message", "")
        reply, agent_reply = chat_turn(user_msg)
        if agent_reply:
            reply += f"\n\n🎁 추천 상품:\n{agent_reply}"
        return JsonResponse({"reply": reply})

