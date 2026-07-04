document.addEventListener("DOMContentLoaded", () => {
  const CYCLE_MS = 16000;

  const presentations = document.querySelectorAll(".presentation");

  presentations.forEach((presentation) => {
    const slides = Array.from(
      presentation.querySelectorAll(".hero-media .hero-snapshot")
    );
    const texts = Array.from(
      presentation.querySelectorAll(".hero-captions .hero-text")
    );
    const buttons = Array.from(
      presentation.querySelectorAll(".hero-buttons button")
    );

    if (
      !slides.length ||
      slides.length !== texts.length ||
      slides.length !== buttons.length
    ) {
      return;
    }

    const count = slides.length;
    const stepMs = CYCLE_MS / count;
    let currentIndex = 0;
    let intervalId;

    const setSlide = (index) => {
      currentIndex = ((index % count) + count) % count;

      slides.forEach((slide, i) => {
        slide.classList.toggle("is-active", i === currentIndex);
      });

      texts.forEach((text, i) => {
        text.classList.toggle("is-active", i === currentIndex);
      });

      buttons.forEach((button, i) => {
        const active = i === currentIndex;
        button.classList.toggle("active", active);
        button.setAttribute("aria-pressed", String(active));
      });
    };

    const startLoop = () => {
      clearInterval(intervalId);
      intervalId = window.setInterval(() => {
        setSlide(currentIndex + 1);
      }, stepMs);
    };

    buttons.forEach((button, index) => {
      button.addEventListener("click", () => {
        setSlide(index);
        startLoop(); // continue loop from clicked slide
      });
    });

    setSlide(0);
    startLoop();
  });
});