document.addEventListener("DOMContentLoaded", function () {
  const slider = document.getElementById('slider');
  const slides = document.querySelectorAll('.slider-wrapper .slide-content');
  const dots = document.querySelectorAll('.slide-indicator .dot');
  const rightSlide = document.querySelector('.right-slide');
  const slideCount = slides.length; // 원본 슬라이드 개수 (예: 5)

  let currentIndex = 0;
  let isTransitioning = false;
  let slideWidth; // 각 슬라이드의 너비 (픽셀 단위)
  let intervalId;

  // ✅ 첫 번째 슬라이드 복제하여 마지막에 추가
  const cloneFirst = slides[0].cloneNode(true);
  slider.appendChild(cloneFirst);

  // ✅ 슬라이더 너비 계산 및 갱신 함수
  function updateSliderWidth() {
    slideWidth = rightSlide.offsetWidth; // .right-slide의 현재 계산된 너비를 가져옴

    console.log("--- updateSliderWidth called ---");
    console.log("rightSlide.offsetWidth (기준 슬라이드 너비):", slideWidth, "px");
    console.log("slideCount (원본):", slideCount);

    // .slider-wrapper의 총 너비를 설정합니다.
    slider.style.width = `${(slideCount + 1) * slideWidth}px`;

    console.log("Calculated slider.style.width (wrapper 총 너비):", slider.style.width);
    console.log("Current index:", currentIndex);

    goToSlide(currentIndex, false); // 너비 변경 시 현재 위치 재조정 (애니메이션 없이)
  }

  // ✅ 슬라이드 이동 함수
  function goToSlide(index, useTransition = true) {
    if (isTransitioning && useTransition) { // 트랜지션 중일 때만 건너뛰도록 조건 수정
      console.log("Transition in progress, skipping goToSlide.");
      return;
    }
    isTransitioning = true;

    console.log(`goToSlide(${index}) called. slideWidth: ${slideWidth}px`);
    console.log(`Target transform: translateX(-${slideWidth * index}px)`);

    slider.style.transition = useTransition ? 'transform 1s ease' : 'none';
    slider.style.transform = `translateX(-${slideWidth * index}px)`;

    dots.forEach(dot => dot.classList.remove('active'));
    dots[index % slideCount].classList.add('active');

    if (index === slideCount) {
      console.log("Reached cloned slide. Resetting to first slide (index 0) after transition.");
      setTimeout(() => {
        slider.style.transition = 'none';
        slider.style.transform = 'translateX(0px)';
        currentIndex = 0;
        isTransitioning = false;
      }, 1000);
    } else {
      setTimeout(() => {
        isTransitioning = false;
      }, 1000);
    }
  }

  // ✅ 자동 슬라이드 이동 함수
  function autoSlide() {
    currentIndex++;
    goToSlide(currentIndex);
  }

  // ✅ dot 클릭 시 해당 슬라이드로 이동
  dots.forEach((dot, i) => {
    dot.addEventListener('click', () => {
      clearInterval(intervalId);
      currentIndex = i;
      goToSlide(currentIndex);
      intervalId = setInterval(autoSlide, 5000);
    });
  });

  // ✅ 리사이즈 대응
  window.addEventListener('resize', () => {
    updateSliderWidth();
  });

  // ✅ 초기화
  updateSliderWidth();
  goToSlide(currentIndex, false);
  intervalId = setInterval(autoSlide, 5000);
});



// document.addEventListener("DOMContentLoaded", function () {
//   const slider = document.getElementById('slider');
//   const slides = document.querySelectorAll('.slider-wrapper .slide-content');
//   const dots = document.querySelectorAll('.slide-indicator .dot');
//   const slideCount = slides.length;

//   let currentIndex = 0;
//   let isTransitioning = false;
//   let slideWidth = document.querySelector('.right-slide').offsetWidth;
//   let intervalId;

//   const cloneFirst = slides[0].cloneNode(true);
//   slider.appendChild(cloneFirst);

//   function updateSliderWidth() {
//     slideWidth = document.querySelector('.right-slide').offsetWidth;
//     slider.style.width = `${(slideCount + 1) * slideWidth}px`;

//     const allSlides = document.querySelectorAll('.slider-wrapper .slide-content');
//     allSlides.forEach(slide => {
//       slide.style.width = `${slideWidth}px`;
//     });

//     goToSlide(currentIndex, false);
//   }

//   function goToSlide(index, useTransition = true) {
//     if (isTransitioning) return;
//     isTransitioning = true;

//     if (useTransition) {
//       slider.style.transition = 'transform 1s ease';
//     } else {
//       slider.style.transition = 'none';
//     }

//     slider.style.transform = `translateX(-${slideWidth * index}px)`;

//     dots.forEach(dot => dot.classList.remove('active'));
//     dots[index % slideCount].classList.add('active');

//     if (index === slideCount) {
//       setTimeout(() => {
//         slider.style.transition = 'none';
//         slider.style.transform = 'translateX(0px)';
//         currentIndex = 0;
//         isTransitioning = false;
//       }, 1000);
//     } else {
//       setTimeout(() => {
//         isTransitioning = false;
//       }, 1000);
//     }
//   }

//   function autoSlide() {
//     currentIndex++;
//     goToSlide(currentIndex);
//   }

//   dots.forEach((dot, i) => {
//     dot.addEventListener('click', () => {
//       clearInterval(intervalId); 
//       currentIndex = i;
//       goToSlide(currentIndex);
//       intervalId = setInterval(autoSlide, 5000); 
//     });
//   });

//   window.addEventListener('resize', () => {
//     updateSliderWidth();
//   });

//   updateSliderWidth();
//   goToSlide(currentIndex, false);
//   intervalId = setInterval(autoSlide, 5000);
// });




// // document.addEventListener("DOMContentLoaded", function () {
// //   const slider = document.getElementById('slider');
// //   const slides = document.querySelectorAll('.slider-wrapper .slide-content');
// //   const dots   = document.querySelectorAll('.slide-indicator .dot');
// //   const slideCount = slides.length;
// //   const slideWidth = document.querySelector('.right-slide').offsetWidth;

// //   let currentIndex = 0;
// //   let isTransitioning = false;

// //   const cloneFirst = slides[0].cloneNode(true);
// //   slider.appendChild(cloneFirst);

// //   slider.style.width = `${(slideCount + 1) * 100}vw`;

// //   function goToSlide(index) {
// //     if (isTransitioning) return;
// //     isTransitioning = true;

// //     slider.style.transition = 'transform 1s ease';
// //     slider.style.transform = `translateX(-${slideWidth * index}px)`;

// //     dots.forEach(dot => dot.classList.remove('active'));
// //     dots[index % slideCount].classList.add('active');

// //     if (index === slideCount) {
// //       setTimeout(() => {
// //         slider.style.transition = 'none'; 
// //         slider.style.transform = 'translateX(0px)';
// //         currentIndex = 0;
// //         isTransitioning = false;
// //       }, 1000); 
// //     } else {
// //       setTimeout(() => {
// //         isTransitioning = false;
// //       }, 1000);
// //     }
// //   }

// //   function autoSlide() {
// //     currentIndex++;
// //     goToSlide(currentIndex);
// //   }

// //   goToSlide(currentIndex);
// //   setInterval(autoSlide, 5000);
// // });
