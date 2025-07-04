import json
from django.utils import timezone
from django.http import JsonResponse
from django.shortcuts import render, redirect
from app.models import User, UserPrefer, PreferType, Chat
from django.views.decorators.csrf import csrf_protect, csrf_exempt
from django.contrib.auth.hashers import make_password, check_password
from allauth.socialaccount.models import SocialAccount
from app.utils import upload_to_s3

def home(request):
    if not request.session.get('user_id') or request.session.get('type') == 'guest':
        return redirect('login')
    
    user_id = request.session.get('user_id')
    try:
        profile = User.objects.get(user_id=user_id)
    except User.DoesNotExist:
        return redirect('login')

    preferences = UserPrefer.objects.filter(user=profile).select_related('prefer_type')
    prefer_tags = [p.prefer_type.type_name for p in preferences]
    
    history_data = []  
    chats = Chat.objects.filter(user_id=profile).select_related('recipient').prefetch_related(
        'chatrecommend_set__product_id'
    ).order_by('-created_at')

    for chat in chats:
        recipient = chat.recipient

        # 추천 상품 목록
        recommends = chat.chatrecommend_set.filter(product_id__isnull=False, is_liked=True)
        products = [
            {
                'rcmd_id': rec.rcmd_id, 
                'brand': rec.product_id.brand,
                'title': rec.product_id.name,
                'imageUrl': rec.product_id.image_url,
                'link': rec.product_id.product_url,
                'is_liked': rec.is_liked,
            }
            for rec in recommends if rec.product_id
        ]
        if products:
            history_data.append((chat, recipient, products))
    return render(request, 'mypage.html', {
        'profile': profile,
        'prefer_tags': prefer_tags,
        'history_data': history_data,
    })

@csrf_protect
def profile_info(request):
    user_id = request.session.get("user_id")
    if not user_id:
        return redirect("login")        
    try:
        user = User.objects.get(user_id=user_id)
    except User.DoesNotExist:
        return redirect("login")
    
    if request.method == "GET":
        preferences = UserPrefer.objects.filter(user=user).select_related("prefer_type")
        style_ids = [p.prefer_type.prefer_id for p in preferences if p.prefer_type.type == "S"]
        category_ids = [p.prefer_type.prefer_id for p in preferences if p.prefer_type.type == "C"]
        style_options = PreferType.objects.filter(type="S")
        category_options = PreferType.objects.filter(type="C")

        return render(request, "profile/profile_info.html", {
            "user": user,
            "style_ids": style_ids,
            "category_ids": category_ids,
            "style_options": style_options,
            "category_options": category_options,
        })

    if request.method == "POST":
        password = request.POST.get("password", "").strip()
        nickname = request.POST.get("nickname", "").strip()
        birth = request.POST.get("birth", "").strip()
        job = request.POST.get("job", "").strip()
        delete_image = request.POST.get("delete_image", "")  # "1"이면 삭제
        style_ids = request.POST.getlist("style")
        category_ids = request.POST.getlist("category")

        if request.session.get("type") != "social" and not check_password(password, user.password):
            preferences = UserPrefer.objects.filter(user=user).select_related("prefer_type")
            style_ids = [p.prefer_type.prefer_id for p in preferences if p.prefer_type.type == "S"]
            category_ids = [p.prefer_type.prefer_id for p in preferences if p.prefer_type.type == "C"]
            style_options = PreferType.objects.filter(type="S")
            category_options = PreferType.objects.filter(type="C")

            return render(request, "profile/profile_info.html", {
                "user": user,
                "style_ids": style_ids,
                "category_ids": category_ids,
                "style_options": style_options,
                "category_options": category_options,
                "error": "비밀번호가 올바르지 않습니다.",
            })

        # 프로필 이미지 업로드 or 삭제 처리
        uploaded_file = request.FILES.get("profile_image")
        if uploaded_file:
            path = upload_to_s3(uploaded_file)  # S3에 업로드   
            if path:
                user.profile_image = path
            else:
                preferences = UserPrefer.objects.filter(user=user).select_related("prefer_type")
                style_ids = [p.prefer_type.prefer_id for p in preferences if p.prefer_type.type == "S"]
                category_ids = [p.prefer_type.prefer_id for p in preferences if p.prefer_type.type == "C"]
                style_options = PreferType.objects.filter(type="S")
                category_options = PreferType.objects.filter(type="C")
                return render(request, "profile/profile_info.html", {
                    "user": user,
                    "style_ids": style_ids,
                    "category_ids": category_ids,
                    "style_options": style_options,
                    "category_options": category_options,
                    "error": "이미지 업로드에 실패했습니다. 다시 시도해주세요.",
                })

        elif delete_image == "1":
            user.profile_image = ""

        # 정보 저장
        user.nickname = nickname
        user.birth = birth
        user.job = job
        user.save()
        
        request.session["nickname"] = user.nickname
        request.session["birth"] = user.birth
        request.session["profile_image"] = user.profile_image or ""
        birth = request.session.get("birth", None)
        if birth:
            # 오늘 날짜 → 'MMDD'만 추출
            today_mmdd = timezone.now().strftime('%m%d')

            # 생일에서 'MMDD'만 추출
            birth_mmdd = birth[4:] if birth else ''

            is_birth_today = (birth_mmdd == today_mmdd)
            request.session['is_birth'] = is_birth_today  # 세션에 저장

        # 선호 태그 갱신
        UserPrefer.objects.filter(user=user).delete()
        for pid in style_ids + category_ids:
            try:
                pref = PreferType.objects.get(prefer_id=int(pid))
                UserPrefer.objects.create(user=user, prefer_type=pref)
            except PreferType.DoesNotExist:
                continue

        return redirect("mypage")

@csrf_exempt
def profile_password(request):
    if request.method == "POST":
        current_password = request.POST.get("current_password", "").strip()
        new_password = request.POST.get("new_password", "").strip()
        confirm_password = request.POST.get("confirm_password", "").strip()
        user_id = request.session.get("user_id")
        if not user_id:
            return redirect("login")

        try:
            user = User.objects.get(user_id=user_id)
        except User.DoesNotExist:
            return redirect("login")

        if not check_password(current_password, user.password):
            return render(request, 'profile/profile_password.html', {
                'error': '현재 비밀번호가 올바르지 않습니다.'
            })

        if new_password != confirm_password:
            return render(request, 'profile/profile_password.html', {
                'error': '새 비밀번호와 확인 비밀번호가 일치하지 않습니다.'
            })
        user.password = make_password(new_password)
        user.save()
        
        # 비밀번호 변경 후 세션 초기화
        request.session.flush()
        request.session["nickname"] = "게스트"
        
        return redirect('profile_password_confirm')
    return render(request, 'profile/profile_password.html')

@csrf_exempt
def password_check(request):
    if request.method == "POST":
        data = json.loads(request.body)
        current_password = data.get("password", "").strip()
        user_id = request.session.get("user_id")
        if not user_id:
            return redirect("login")

        try:
            user = User.objects.get(user_id=user_id)
        except User.DoesNotExist:
            return redirect("login")

        if not check_password(current_password, user.password):
            return JsonResponse({"success": False, "message": "현재 비밀번호가 올바르지 않습니다."}, status=400)
        return JsonResponse({"success": True, "message": "비밀번호가 확인되었습니다."})

def profile_password_confirm(request):
    return render(request, 'profile/profile_password_confirm.html')

def profile_delete(request):
    return render(request, 'profile/profile_delete.html')

def profile_delete_confirm(request):
    return render(request, 'profile/profile_delete_confirm.html')

@csrf_exempt
def delete_user_account(request):
    if request.method != "POST":
        return JsonResponse({"success": False, "message": "잘못된 요청입니다."}, status=400)

    user_id = request.session.get("user_id")
    if not user_id:
        return JsonResponse({"success": False, "message": "로그인이 필요합니다."}, status=401)

    try:
        user = User.objects.get(user_id=user_id)
    except User.DoesNotExist:
        return JsonResponse({"success": False, "message": "사용자 없음"}, status=404)

    try:
        data = json.loads(request.body)
        reason = data.get("reason", "").strip()
        if not reason:
            return JsonResponse({"success": False, "message": "탈퇴 사유가 필요합니다."}, status=400)

        # 사용자 정보 수정
        user.reason = reason
        user.deleted_at = timezone.now()

        user.email = f"deleted_{user_id}_{user.email}"
        user.save()
        
        user = request.user
        # SocialAccount 삭제
        SocialAccount.objects.filter(user=user).delete()

        # 로그아웃 처리
        request.session.flush()
        request.session["nickname"] = "게스트"

        return JsonResponse({"success": True, "message": "계정이 성공적으로 삭제되었습니다."}, status=200)
    except Exception as e:
        return JsonResponse({"success": False, "message": str(e)}, status=400)