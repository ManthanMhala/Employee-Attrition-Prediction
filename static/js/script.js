document.addEventListener("DOMContentLoaded", () => {
  const menuToggle = document.getElementById("menuToggle");
  const sidebar = document.getElementById("sidebar");

  if (menuToggle && sidebar) {
    menuToggle.addEventListener("click", () => {
      sidebar.classList.toggle("sidebar-open");
      document.body.classList.toggle("nav-open", sidebar.classList.contains("sidebar-open"));
    });
    document.addEventListener("click", (event) => {
      if (window.innerWidth <= 900 && sidebar.classList.contains("sidebar-open") &&
          !sidebar.contains(event.target) && !menuToggle.contains(event.target)) {
        sidebar.classList.remove("sidebar-open");
        document.body.classList.remove("nav-open");
      }
    });
  }

  document.querySelectorAll(".flash-close").forEach((button) => {
    button.addEventListener("click", () => button.closest(".flash-message")?.remove());
  });

  document.querySelectorAll("form[data-confirm]").forEach((form) => {
    form.addEventListener("submit", (event) => {
      if (!window.confirm(form.dataset.confirm || "Are you sure?")) event.preventDefault();
    });
  });

  document.querySelectorAll(".chart-container .plotly-graph-div").forEach((chart) => {
    if (window.Plotly) {
      window.Plotly.Plots.resize(chart);
      window.addEventListener("resize", () => window.Plotly.Plots.resize(chart));
    }
  });

  const predictionForm = document.getElementById("predictionForm");
  if (predictionForm) {
    predictionForm.addEventListener("submit", () => {
      const submitButton = predictionForm.querySelector('button[type="submit"]');
      if (submitButton) {
        submitButton.disabled = true;
        submitButton.dataset.originalText = submitButton.innerText;
        submitButton.innerText = "Generating assessment…";
      }
    });
  }
});
