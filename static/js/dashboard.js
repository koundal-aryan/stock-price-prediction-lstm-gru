/**
 * ADANIPORTS Stock Price Prediction Dashboard
 * Handles dynamic chart resizing, UI interactions, and interactive helpers.
 */

document.addEventListener("DOMContentLoaded", function () {
    // Responsive Plotly chart redraw on window resize
    window.addEventListener("resize", function () {
        const plotContainers = document.querySelectorAll(".js-plotly-plot");
        plotContainers.forEach((plot) => {
            if (window.Plotly) {
                window.Plotly.Plots.resize(plot);
            }
        });
    });

    // Auto-scroll smooth behavior for internal links
    document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
        anchor.addEventListener("click", function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute("href"));
            if (target) {
                target.scrollIntoView({ behavior: "smooth" });
            }
        });
    });
});
