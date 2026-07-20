import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# --- 1. Python Calculation ---
X = np.array([
    [2, 70, 12],
    [4, 80, 15],
    [6, 85, 18],
    [8, 90, 20],
    [10, 95, 22]
])
y = np.array([45, 60, 75, 90, 98])

model = LinearRegression()
model.fit(X, y)

intercept = model.intercept_
coefs = model.coef_
r2 = model.score(X, y)

# --- 2. Generate Graphs & Convert to Base64 ---
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

study_hours = X[:, 0]
attendance = X[:, 1]
assignment = X[:, 2]
preds = model.predict(X)

# Chart 1: Actual vs Predicted
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(y, preds, color='black', label='Model Predictions')
ax.plot([40, 100], [40, 100], color='gray', linestyle='--', label='Perfect Fit (y=x)')
ax.set_xlabel('Actual Marks')
ax.set_ylabel('Predicted Marks')
ax.set_title('Actual vs Predicted Final Marks')
ax.legend()
img1 = fig_to_base64(fig)

# Chart 2: Study Hours vs Final Marks
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(study_hours, y, color='black')
ax.set_xlabel('Study Hours')
ax.set_ylabel('Final Marks')
ax.set_title('Study Hours vs Final Marks')
img2 = fig_to_base64(fig)

# Chart 3: Assignment Score vs Final Marks
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(assignment, y, color='black')
ax.set_xlabel('Assignment Score')
ax.set_ylabel('Final Marks')
ax.set_title('Assignment Score vs Final Marks')
img3 = fig_to_base64(fig)


# --- 3. Standalone HTML Content ---
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multiple Linear Regression Predictor</title>
    <!-- PyScript to run Python directly in the browser with zero JS and no server -->
    <link rel="stylesheet" href="https://pyscript.net/releases/2024.1.1/core.css">
    <script type="module" src="https://pyscript.net/releases/2024.1.1/core.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #ffffff;
            color: #000000;
            margin: 0;
            padding: 40px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            border: 1px solid #000000;
            padding: 30px;
        }}
        .section {{
            margin: 30px 0;
        }}
        .form-group {{
            margin: 15px 0;
        }}
        label {{
            display: inline-block;
            width: 180px;
            font-weight: bold;
        }}
        input[type="number"] {{
            padding: 6px;
            width: 120px;
            border: 1px solid #000000;
        }}
        button {{
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #000000;
            padding: 8px 16px;
            cursor: pointer;
            font-weight: bold;
            margin-top: 10px;
        }}
        button:hover {{
            background-color: #000000;
            color: #ffffff;
        }}
        .equation-box {{
            border: 1px solid #000000;
            padding: 15px;
            background-color: #f9f9f9;
            margin: 15px 0;
            font-family: monospace;
        }}
        .result-box {{
            border: 1px solid #000000;
            padding: 15px;
            background-color: #f0f0f0;
            margin: 15px 0;
            font-weight: bold;
            font-size: 1.1em;
        }}
        /* CSS-only toggle for graphs (Zero JavaScript) */
        #toggle-graphs {{
            display: none;
        }}
        .toggle-btn {{
            display: inline-block;
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #000000;
            padding: 8px 16px;
            cursor: pointer;
            font-weight: bold;
            text-align: center;
        }}
        .toggle-btn:hover {{
            background-color: #000000;
            color: #ffffff;
        }}
        .graphs-section {{
            display: none;
            margin-top: 30px;
            text-align: center;
        }}
        #toggle-graphs:checked ~ .graphs-section {{
            display: block;
        }}
        #toggle-graphs:checked ~ div label.toggle-btn::after {{
            content: "Hide Graphs";
        }}
        .toggle-btn::after {{
            content: "Show Graphs";
        }}
        .chart-img {{
            max-width: 100%;
            border: 1px solid #000000;
            margin: 15px auto;
        }}
        .charts-grid {{
            display: flex;
            gap: 20px;
            justify-content: center;
            flex-wrap: wrap;
        }}
        .charts-grid img {{
            max-width: 48%;
        }}
    </style>
</head>
<body>

<div class="container">
    <div class="section">
        <p>The Multiple Linear Regression model equation is:</p>
        <div class="equation-box">
            Final Marks = {intercept:.2f} + ({coefs[0]:.2f} × Study Hours) + ({coefs[1]:.2f} × Attendance) + ({coefs[2]:.2f} × Assignment)
        </div>
        <ul>
            <li>Intercept (β₀): {intercept:.4f}</li>
            <li>Study Hours Coefficient (β₁): {coefs[0]:.4f}</li>
            <li>Attendance Coefficient (β₂): {coefs[1]:.4f}</li>
            <li>Assignment Coefficient (β₃): {coefs[2]:.4f}</li>
            <li>R-squared (R²): {r2:.4f}</li>
        </ul>
    </div>

    <!-- Prediction Form with Input Fields processed by Python -->
    <div class="section">
        <div class="form-group">
            <label for="studyHours">Study Hours:</label>
            <input type="number" id="studyHours" value="5" step="0.1" min="0">
        </div>
        <div class="form-group">
            <label for="attendance">Attendance (%):</label>
            <input type="number" id="attendance" value="82" step="0.1" min="0" max="100">
        </div>
        <div class="form-group">
            <label for="assignment">Assignment Score:</label>
            <input type="number" id="assignment" value="16" step="0.1" min="0">
        </div>
        <button py-click="calculate_prediction">Predict Final Marks</button>
        <div class="result-box" id="predictionResult">
            Predicted Final Marks: --
        </div>
    </div>

    <!-- CSS-only Toggle for Graphs -->
    <div class="section" style="text-align: center;">
        <input type="checkbox" id="toggle-graphs">
        <div>
            <label for="toggle-graphs" class="toggle-btn"></label>
        </div>
        
        <div class="graphs-section">
            <div style="margin-bottom: 20px;">
                <img src="data:image/png;base64,{img1}" alt="Actual vs Predicted" class="chart-img">
            </div>
            <div class="charts-grid">
                <img src="data:image/png;base64,{img2}" alt="Study Hours vs Marks" class="chart-img">
                <img src="data:image/png;base64,{img3}" alt="Assignment vs Marks" class="chart-img">
            </div>
        </div>
    </div>
</div>

<!-- Python code running directly in the website via PyScript -->
<script type="py">
from pyscript import document

def calculate_prediction(event):
    try:
        sh = float(document.querySelector("#studyHours").value)
        att = float(document.querySelector("#attendance").value)
        ass = float(document.querySelector("#assignment").value)
        
        # Multiple Linear Regression formula computed by Python
        intercept = {intercept}
        coef_study = {coefs[0]}
        coef_att = {coefs[1]}
        coef_ass = {coefs[2]}
        
        pred = intercept + (coef_study * sh) + (coef_att * att) + (coef_ass * ass)
        
        document.querySelector("#predictionResult").innerText = f"Predicted Final Marks: {{pred:.2f}}"
    except Exception as e:
        document.querySelector("#predictionResult").innerText = f"Error: {{e}}"
</script>

</body>
</html>
"""

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("index.html successfully generated with base64 embedded graphs!")
