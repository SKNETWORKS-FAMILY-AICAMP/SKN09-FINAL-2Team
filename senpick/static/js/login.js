document.addEventListener("DOMContentLoaded", function () {
  const emailInput = document.getElementById("email");
  const emailError = document.getElementById("email-error");
  const pwInput = document.getElementById("password");
  const pwError = document.getElementById("password-error");
  const loginForm = document.querySelector(".login-form");

  function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(";").shift();
  }

  emailInput.addEventListener("focus", () => {
    if (emailInput.value.trim() === "") {
      emailInput.classList.add("error");
      // emailError.textContent = "이메일을 입력해주세요.";
      // emailError.style.display = "block";
      emailError.style.visibility = "visible";
    }
  });

  pwInput.addEventListener("focus", () => {
    if (pwInput.value.trim() === "") {
      pwInput.classList.add("error");
      // pwError.textContent = "비밀번호를 입력해주세요.";
      // pwError.style.display = "block";
      pwError.style.visibility = "visible";
    }
  });

  emailInput.addEventListener("blur", () => {
    if (emailInput.value.trim() !== "") {
      emailInput.classList.remove("error");
      emailError.textContent = ""; // 에러 메시지 초기화
      emailError.style.visibility = "hidden";
    }
  });

  pwInput.addEventListener("blur", () => {
    if (pwInput.value.trim() !== "") {
      pwInput.classList.remove("error");
      pwError.textContent = ""; // 에러 메시지 초기화
      pwError.style.visibility = "hidden";
    }
  });

  loginForm.addEventListener("submit", function (e) {
    e.preventDefault();

    const email = emailInput.value.trim();
    const password = pwInput.value.trim();

    emailInput.classList.remove("error");
    emailError.style.display = "none";
    pwInput.classList.remove("error");
    pwError.style.display = "none";

    let valid = true;

    if (!email) {
      emailInput.classList.add("error");
      // emailError.textContent = "이메일을 입력해주세요.";
      // emailError.style.display = "block";
      // valid = false;
      emailError.textContent = "이메일이 올바르지 않습니다.";
      emailError.style.visibility = "visible";
      valid = false;
    } else {
      emailInput.classList.remove("error");
      emailError.textContent = ""; // 에러 메시지 초기화
      emailError.style.visibility = "hidden";
    }

    if (!password) {
      pwInput.classList.add("error");
      // pwError.textContent = "비밀번호를 입력해주세요.";
      // pwError.style.display = "block";
      // valid = false;
      pwError.textContent = "비밀번호가 올바르지 않습니다.";
      pwError.style.visibility = "visible";
      valid = false;
    } else {
      pwInput.classList.remove("error");
      pwError.textContent = ""; // 에러 메시지 초기화
      pwError.style.visibility = "hidden";
    }

    if (!valid) return;

    // 서버로 AJAX 요청
    fetch("/login/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCookie("csrftoken")
      },
      body: JSON.stringify({
        username: email,
        password: password
      })
    })
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          window.location.href = "/chat";
        } else {
          if (data.email_error) {
            emailInput.classList.add("error");
            emailError.textContent = data.email_error;
            emailError.style.display = "block";
          }
          if (data.password_error) {
            pwInput.classList.add("error");
            pwError.textContent = data.password_error;
            pwError.style.display = "block";
          }
        }
      })
      .catch(err => {
        console.error("로그인 요청 실패", err);
      });
  });
});
