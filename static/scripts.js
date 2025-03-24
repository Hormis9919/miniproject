document.addEventListener("DOMContentLoaded", function () {
    console.log("JavaScript Loaded ✅");

    document.getElementById("predictionForm").onsubmit = async function (event) {
        event.preventDefault();
        let formData = new FormData(this);

        try {
            let response = await fetch("/predict", {
                method: "POST",
                body: formData
            });
            let result = await response.json();

            document.getElementById("result").textContent = result.prediction
            ? `Predicted Price: ₹${result.prediction.toLocaleString()}`
            : `Error: ${result.error}`;
        } catch (error) {
            document.getElementById("result").textContent = "Error fetching prediction ❌";
        }
    };
});
