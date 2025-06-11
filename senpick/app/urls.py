"""
URL configuration for senpick project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path, include
from . import pswd_views
from . import pswd_verif_views
from . import pswd_gen_views
from . import views, birth_views, login_views, chat_views, mypage_views, recommend_views
# from app.views import user_views, chat_views, recommend_views
from app.views import social_redirect_view
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', chat_views.chat, name='chat'),

    path('login/', login_views.login_view, name='login'),                                                # 로그인
    path('logout/', login_views.logout_view, name='logout'),                                             # 로그아웃
    path('pswd/', pswd_views.home, name='pswd'),                                                         # 비밀번호 찾기
    path('pswd/verif/', pswd_verif_views.home, name='pswd_verif'),                                       # 이메일 인증
    
    path("api/pswd_request/", pswd_views.password_reset_request, name="password_reset_request"),         # 인증번호 전송
    path("api/resend_code/", pswd_verif_views.resend_verification_code, name="resend_code"),             # 재전송
    path("api/verify_code/", pswd_verif_views.verify_code, name="verify_code"),                          # 유효성 검사
    path("pswd/gen/", pswd_gen_views.home, name="pswd_gen"),                                             # 비밀번호 생성 
    path("api/set_password/", pswd_gen_views.set_new_password, name="set_password"),
    path("api/check_password/", mypage_views.password_check, name="check_password"),
    
    path('birth/', birth_views.birth, name='birth'),  # 생일 페이지
    path('birth/recommend-products/', birth_views.birth_recommend_products, name='birth_recommend_products'),  # 생일 추천 상품 페이지
    
    path('chat/', chat_views.chat, name='chat'),
    path('chat/<int:chat_id>/', chat_views.chat_detail, name='chat_detail'),  # 채팅 상세 페이지
    path('chat/guest-start/', chat_views.chat_guest_start, name='chat_guest_start'),
    
    path('mypage/', mypage_views.home, name='mypage'),
    path('mypage/profile/', mypage_views.profile_info, name='profile_info'),
    path('mypage/profile/password/', mypage_views.profile_password, name='profile_password'),
    path('mypage/profile/password/confirm', mypage_views.profile_password_confirm, name='profile_password_confirm'),
    path('mypage/profile/delete/', mypage_views.profile_delete, name='profile_delete'),
    path('mypage/profile/delete/confirm/', mypage_views.profile_delete_confirm, name='profile_delete_confirm'),
    path("api/profile/delete/", mypage_views.delete_user_account, name="delete_user_account"),
    
    path('signup/step1/', views.signup_step1, name='signup_step1'),
    path('signup/step2/', views.signup_step2, name='signup_step2'),
    path('signup/step3/', views.signup_step3, name='signup_step3'),
    path('signup/step4/', views.signup_step4, name='signup_step4'),
    path('signup/step5/', views.signup_step5, name='signup_step5'),
    path('signup/send-code/', views.send_verification_code, name='send_verification_code'),
    path('signup/verify-code/', views.verify_code, name='verify_code'),
    path('signup/check-dup/', views.check_duplicate, name='check_duplicate'),
    
    path('accounts/', include('allauth.urls')),
    path('social/redirect/', social_redirect_view, name='social_redirect'),

    path('chat/start/', chat_views.chat_start, name='chat_start'),
    path('chat/message/', chat_views.chat_message, name='chat_message'),
    path('chat/history/', chat_views.chat_history, name='chat_history'),
    path('chat/upload/', chat_views.chat_upload, name='chat_upload'),
    path('chat/feedback/<int:msg_id>', chat_views.chat_feedback, name='chat_feedback'),
    
    path('recommends', recommend_views.index, name='recommends'),  # 추천 상품 조회
    path('recommends/<int:recommend_id>/like', recommend_views.like, name='recommend_like'),  # 추천 상품 좋아요
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
