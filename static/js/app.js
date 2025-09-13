// API client class to handle requests
class PredictionAPI {
    constructor(baseURL = 'http://localhost:5000') {
        this.baseURL = baseURL;
    }

    async predictFertilizer(formData) {
        try {
            const response = await fetch(`${this.baseURL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || 'Fertilizer prediction failed');
            }
            return await response.json();
        } catch (error) {
            console.error('Fertilizer prediction error:', error);
            throw error;
        }
    }

    async predictCrop(formData) {
        try {
            const response = await fetch(`${this.baseURL}/predict-crop`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || 'Crop prediction failed');
            }
            return await response.json();
        } catch (error) {
            console.error('Crop prediction error:', error);
            throw error;
        }
    }

    async predictYield(formData){
        try{
            const response = await fetch(`${this.baseURL}/predict_yield`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(formData)
            });
            if (!response.ok){
                const errorData = await response.json();
                throw new Error(errorData.message || "Yield prediction failed");
            }
            return await response.json();
        } catch (error) {
            console.error("Yield prediction error: ", error);
            throw error;
        }
    }

    async healthCheck() {
        try {
            const response = await fetch(`${this.baseURL}/health`);
            return await response.json();
        } catch (error) {
            console.error('Health check failed:', error);
            return { status: 'error', model_loaded: false };
        }
    }
}

// Initialize API client
const api = new PredictionAPI();

// Utility functions to show/hide loading and display results/errors
function showLoading() {
    document.getElementById('loading').style.display = 'block';
    // Keep buttons visible during loading
    // document.getElementById('fertilizerRecommender').style.display = 'none';
    // document.getElementById('cropRecommender').style.display = 'none';
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

function displayFertilizerResults(result) {
    const resultsDiv = document.getElementById('fertilizerResults');
    resultsDiv.innerHTML = `
        <h3>Fertilizer Recommendation</h3>
        <p><strong>Recommended Fertilizer:</strong> ${result.fertilizer}</p>
        <p><strong>Confidence:</strong> ${result.confidence}%</p>
    `;
    resultsDiv.style.display = 'block';
}

function displayFertilizerError(message) {
    const resultsDiv = document.getElementById('fertilizerResults');
    resultsDiv.innerHTML = `<p class="error">Error: ${message}</p>`;
    resultsDiv.style.display = 'block';
}

function displayCropResults(result) {
    const resultsDiv = document.getElementById('cropResults');
    if (result.recommendations && result.recommendations.length > 0) {
        let html = '<h3>Top 5 Crop Recommendations</h3><table><thead><tr><th>Crop</th><th>Estimated Profit</th><th>Estimated Yield</th><th>Best Season</th></tr></thead><tbody>';
        result.recommendations.forEach(rec => {
            html += `<tr>
                <td>${rec.crop}</td>
                <td>${rec.profit}</td>
                <td>${rec.yield}</td>
                <td>${rec.season}</td>
            </tr>`;
        });
        html += '</tbody></table>';
        resultsDiv.innerHTML = html;
    } else {
        resultsDiv.innerHTML = '<p>No crop recommendations found.</p>';
    }
    resultsDiv.style.display = 'block';
}

function displayCropError(message) {
    const resultsDiv = document.getElementById('cropResults');
    resultsDiv.innerHTML = `<p class="error">Error: ${message}</p>`;
    resultsDiv.style.display = 'block';
}

function displayYieldResults(result){
    const resultsDiv = document.getElementById("yieldResults");
    resultsDiv.innerHTML = `
        <h3> Yield Prediction </h3>
        <p><strong>Yield Prediction would be : </strong> ${result.prediction}</p>
    `;
    resultsDiv.style.display="block";
}

function displayYieldError(message){
    const resultsDiv = document.getElementById("yieldResults");
    resultsDiv.innerHTML = `<p class="error"> Error: ${message}</p>`;
    resultsDiv.style.display = "block";
}

// Collect form data from inputs
function collectFormData() {
    return {
        District_Name: document.getElementById('District_Name').value,
        Soil_color: document.getElementById('Soil_color').value,
        Nitrogen: parseFloat(document.getElementById('Nitrogen').value),
        Phosphorus: parseFloat(document.getElementById('Phosphorus').value),
        Potassium: parseFloat(document.getElementById('Potassium').value),
        pH: parseFloat(document.getElementById('pH').value),
        Rainfall: parseFloat(document.getElementById('Rainfall').value),
        Temperature: parseFloat(document.getElementById('Temperature').value),
        Crop: document.getElementById('Crop').value
    };
}

// Validate required fields for both predictions
function validateFormData(formData, typeOfModel) {
    let requiredFields;
    if (typeOfModel === 0){
        requiredFields = ['District_Name', 'Soil_color', 'Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature', 'Crop']
    } else if (typeOfModel === 1) {
        requiredFields = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature'];
    } else if (typeOfModel === 2) {
        requiredFields = ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "pH", "Rainfall", "Soil_color"]
    }
    // const requiredFields = forFertilizer
    //     ? ['District_Name', 'Soil_color', 'Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature', 'Crop']
    //     : ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature'];

    for (const field of requiredFields) {
        const value = formData[field];
        if (value === '' || value === null || value === undefined || (typeof value === 'number' && isNaN(value))) {
            return `Please fill in the required field: ${field}`;
        }
    }
    if (typeOfModel === 0) {
        if (!formData.District_Name) return 'Please select a District';
        if (!formData.Soil_color) return 'Please select a Soil Type';
        if (!formData.Crop) return 'Please select a Crop Type';
    }
    return null;
}

// Fertilizer prediction handler
async function handleFertilizerPrediction() {
    showLoading();
    try {
        const formData = collectFormData();
        const validationError = validateFormData(formData, 0);
        if (validationError) {
            throw new Error(validationError);
        }
        const result = await api.predictFertilizer(formData);
        displayFertilizerResults(result);
    } catch (error) {
        displayFertilizerError(error.message);
    } finally {
        hideLoading();
    }
}

// Crop prediction handler
async function handleCropPrediction() {
    showLoading();
    try {
        const formData = collectFormData();
        const validationError = validateFormData(formData, 1);
        if (validationError) {
            throw new Error(validationError);
        }
        const result = await api.predictCrop(formData);
        displayCropResults(result);
    } catch (error) {
        displayCropError(error.message);
    } finally {
        hideLoading();
    }
}

// Yield prediction handler 
async function handleYieldPrediction() {
    showLoading();
    try{
        const formData = collectFormData();
        const validationError = validateFormData(formData, 2); 
        if (validationError){
            throw new Error(validationError);
        }
        const result = await api.predictYield(formData);
        displayYieldResults(result);
    } catch (error) {
        displayYieldError(error.message);
    } finally {
        hideLoading();
    }
}

// Event listeners setup
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('fertilizerRecommender').addEventListener('click', handleFertilizerPrediction);
    document.getElementById('cropRecommender').addEventListener('click', handleCropPrediction);
    document.getElementById('yieldRecommender').addEventListener('click', handleYieldPrediction); 
});