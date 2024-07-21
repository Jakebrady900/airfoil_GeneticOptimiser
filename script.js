document.addEventListener("DOMContentLoaded", function() {
    function sendHttpRequest(method, url, body = null) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open(method, url);
            xhr.setRequestHeader('Content-Type', 'application/json');

            xhr.onload = function () {
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve(xhr.response);
                } else {
                    reject(new Error('Request failed with status ' + xhr.status));
                }
            };

            xhr.onerror = function () {
                reject(new Error('Request failed'));
            };

            xhr.send(body ? JSON.stringify(body) : null);
        });
    }

    async function pollUntilComplete(endpoint, chart) {
        while (true) {
            try {
                const response = await sendHttpRequest('GET', endpoint);
                const jsonResponse = JSON.parse(response);

                if (jsonResponse.status && jsonResponse.status === "Complete.") {
                    console.log("Process completed successfully!");
                    await fetchAndUpdateImage(jsonResponse.Airfoil.AOA);
                    break;
                } else if (jsonResponse.fitness_tracker) {
                    console.log("Updating chart with new data...");
                    const [index, fitness] = jsonResponse.fitness_tracker.slice(-1)[0]; // Get the latest data point
                    updateChart(chart, index, fitness);
                } else {
                    console.log("Waiting for process completion...");
                }
            } catch (error) {
                console.error("Error:", error);
                break;
            }
            await new Promise(resolve => setTimeout(resolve, 3000)); // Wait for 3 seconds before the next request
        }
    }

    document.getElementById("solve-button").addEventListener("click", async function (event) {
        event.preventDefault(); // Prevent default form submission behavior

        const solutionType = parseInt(document.querySelector('input[name="options"]:checked').value);
        const velocity = parseInt(document.getElementById("velocity").value);

        const payLoad = {
            "solution_type": solutionType,
            "velocity": velocity
        };

        const postUrl = "http://localhost:8081/run";
        const pollUrl = "http://localhost:8081/get_status";

        try {
            await sendHttpRequest('POST', postUrl, payLoad);
            console.log("POST request sent successfully!");

            const ctx = document.getElementById('fitnessChart').getContext('2d');
            const fitnessChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Fitness',
                        data: [],
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1,
                        fill: false
                    }]
                },
                options: {
                    scales: {
                        x: {
                            type: 'linear',
                            position: 'bottom'
                        }
                    }
                }
            });

            await pollUntilComplete(pollUrl, fitnessChart);
        } catch (error) {
            console.error("Error:", error);
        }
    });

    async function fetchAndUpdateImage(rotationDegrees) {
        try {
            const response = await fetch('http://localhost:8081/outputs/airfoil.png'); 
            if (!response.ok) {
                throw new Error('Failed to fetch image');
            }
            const blob = await response.blob();
            const imageURL = URL.createObjectURL(blob);

            document.getElementById('image-container').innerHTML = `
            <img src="${imageURL}" alt="Updated Image" style="transform: rotate(${rotationDegrees}deg);">
            `;
        } catch (error) {
            console.error('Error fetching image:', error);
        }
    }

    function updateChart(chart, index, fitness) {
        chart.data.labels.push(index);
        chart.data.datasets[0].data.push(fitness);
        chart.update();
    }
});
