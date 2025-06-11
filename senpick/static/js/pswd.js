document.addEventListener("DOMContentLoaded", function () {
  const logoBtn = document.querySelector(".logo");
  const emailInput = document.getElementById("email");
  const emailError = document.getElementById("email-error");
  const pswdForm = document.querySelector(".pswd-form");
  console.log("pswdForm loaded:", pswdForm);

  logoBtn.addEventListener("click", () => window.location.href = "/login");

  emailInput.addEventListener("focus", function () {
        // focus 시에는 초기화만 수행하고, 입력 여부 검사는 input 또는 blur에서 처리
        emailInput.classList.remove("error");
        emailError.classList.remove("show");
    });

  emailInput.addEventListener("input", function () {
        if (emailInput.classList.contains("error")) {
            emailInput.classList.remove("error");
        }
        if (emailError.classList.contains("show")) {
            emailError.classList.remove("show");
        }
    });

  emailInput.addEventListener("blur", function () {
        const email = emailInput.value.trim();
        if (email === "") {
            emailInput.classList.add("error");
            emailError.textContent = "이메일을 필수 입력해주세요.";
            emailError.classList.add("show");
        } else {
            const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/; // 기본적인 이메일 정규 표현식
            if (!emailPattern.test(email)) {
                emailInput.classList.add("error");
                emailError.textContent = "유효한 이메일 형식이 아닙니다."; // 커스텀 에러 메시지
                emailError.classList.add("show");
            } else {
                emailInput.classList.remove("error");
                emailError.classList.remove("show");
            }
        }
    });

  loginForm.addEventListener("submit", function (e) {
    e.preventDefault();
    console.log("submit 이벤트 발생");
    const email = emailInput.value.trim();

    if (email === "") {
      emailInput.classList.add("error");
      emailError.textContent = "이메일을 필수 입력해주세요.";
      emailError.classList.add("show");
      return;
    }

    const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailPattern.test(email)) {
            emailInput.classList.add("error");
            emailError.textContent = "유효한 이메일 형식이 아닙니다."; // 커스텀 에러 메시지
            emailError.classList.add("show");
            return; // 유효하지 않으면 폼 제출을 막고 종료
        }

    fetch("/api/pswd_request/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCookie("csrftoken")
      },
      credentials: "include", 
      body: JSON.stringify({ email: email })
    })
      .then((res) => {
        console.log("fetch 응답 status:", res.status);
        return res.json();
  })
      .then((data) => {
        console.log("fetch 결과:", data);
        if (data.success) {
          window.location.href = data.redirect_url;
        } else {
          console.error("서버 응답 실패:", data.message);
          emailInput.classList.add("error");
          emailError.textContent = data.message || "가입되지 않은 이메일입니다.";
          emailError.classList.add("show");
        }
      })
      .catch((err) => {
        console.error("fetch 실패:", err);
        emailInput.classList.add("error");
        emailError.textContent = "서버 요청 중 오류가 발생했습니다.";
        emailError.classList.add("show");
      });
  });

  function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(";").shift();
  }
});
