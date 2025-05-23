from django.urls import path
from .views import chat_respond, chatbot_page

urlpatterns = [
    path("", chatbot_page),  # 👈 /chat/ 로 접근 시 chatbot.html 띄움
    path("respond/", chat_respond, name="chat_respond"),
]
