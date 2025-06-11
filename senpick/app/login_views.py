from django.shortcuts import render, redirect
from django.contrib.auth.hashers import check_password
from app.models.user import User 
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.contrib.auth.hashers import check_password
from django.views.decorators.csrf import csrf_exempt
from app.models.user import User
import json

def home(request):
    return render(request, 'login.html')

def login_view(request):
    if request.method == "GET":
        return render(request, "login.html")

    if request.method == "POST":
        email = request.POST.get("username", "").strip()
        password = request.POST.get("password", "").strip()

        if not email:
            return render(request, "login.html", {"email_error": "이메일이 올바르지 않습니다."})
        if not password:
            return render(request, "login.html", {"password_error": "이메일 또는 비밀번호가 올바르지 않습니다."})

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return render(request, "login.html", {"email_error": "이메일이 올바르지 않습니다."})

        if not check_password(password, user.password):
            return render(request, "login.html", {"password_error": "이메일 또는 비밀번호가 올바르지 않습니다."})

        request.session.flush()  # 세션 초기화  
        request.session["user_id"] = user.user_id
        request.session["nickname"] = user.nickname
        request.session["birth"] = user.birth
        request.session["type"] = user.type
        request.session["profile_image"] = user.profile_image or ""
        
        return redirect("chat")

    return render(request, "login.html", {"password_error": "잘못된 접근입니다."})

def logout_view(request):
    request.session.flush()  # 세션 초기화
    return redirect('login')  # 로그인 페이지로 리다이렉트
