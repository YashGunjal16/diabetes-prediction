from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os

# Initialize Flask app
app = Flask(__name__)

# Load the dataset
data = pd.read_csv("diabetes.csv")

# Features and target variable
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=12345)
rf_model.fit(X_train, y_train)

# Generate PDF report
def generate_pdf_report(user_input, prediction_text, probability, precautionary_text, suggested_meds, filename):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    title = Paragraph("<b>Diabetes Prediction Report</b>", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # User input table
    data = [["Patient Detail", "Value"]]
    for feature, value in user_input.items():
        data.append([feature, value])

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Prediction and probability
    prediction_paragraph = Paragraph(
        f"<b>Prediction:</b> The model predicts {prediction_text} with a probability of {probability[1]:.2f}.",
        styles['Normal']
    )
    elements.append(prediction_paragraph)
    elements.append(Spacer(1, 12))

    # Precautionary notes
    formatted_precautionary_text = precautionary_text.replace('\n', '<br/>')
    precautionary_paragraph = Paragraph(f"<b>Precautionary Notes:</b><br/>{formatted_precautionary_text}", styles['Normal'])
    elements.append(precautionary_paragraph)
    elements.append(Spacer(1, 12))

    # Suggested medications
    formatted_meds = suggested_meds.replace('\n', '<br/>')
    medications_paragraph = Paragraph(f"<b>Suggested Medications:</b><br/>{formatted_meds}", styles['Normal'])
    elements.append(medications_paragraph)

    # Build the PDF
    doc.build(elements)

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        user_input = {
            "Pregnancies": int(data['pregnancies']),
            "Glucose": float(data['glucose']),
            "BloodPressure": float(data['blood_pressure']),
            "SkinThickness": float(data['skin_thickness']),
            "Insulin": float(data['insulin']),
            "BMI": float(data['bmi']),
            "DiabetesPedigreeFunction": float(data['diabetes_pedigree']),
            "Age": int(data['age']),
        }
        user_data = pd.DataFrame([user_input])

        # Prediction and probability
        prediction = rf_model.predict(user_data)[0]
        probability = rf_model.predict_proba(user_data)[0]
        prediction_text = "DIABETES" if prediction == 1 else "NO DIABETES"

        # Precautionary notes and suggested medications
        precautionary_text = (
            "If you are diagnosed with diabetes, here are some precautions:\n"
            "- Follow a healthy diet with low sugar and processed foods.\n"
            "- Monitor blood sugar regularly.\n"
            "- Regular exercise and weight management.\n"
            "- Consult an endocrinologist for further advice."
        ) if prediction == 1 else (
            "If you are not diagnosed with diabetes, here are some precautions:\n"
            "- Maintain a healthy lifestyle with balanced nutrition and exercise.\n"
            "- Regular health checkups, especially blood sugar levels.\n"
            "- Stay hydrated and avoid stress."
        )

        suggested_meds = (
            "Medicines for Diabetes:\n- Metformin\n- Insulin\n- Glipizide\n- Liraglutide"
        ) if prediction == 1 else "No medications required as you don't have diabetes."

        # Generate PDF report
        report_filename = "Diabetes_Prediction_Report.pdf"
        generate_pdf_report(user_input, prediction_text, probability, precautionary_text, suggested_meds, report_filename)

        return jsonify({
            "prediction": prediction_text,
            "probability": probability[1],
            "report_link": "/download_report",
            "precautions": precautionary_text,
            "medications": suggested_meds
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# Download report endpoint
@app.route('/download_report', methods=['GET'])
def download_report():
    report_path = "Diabetes_Prediction_Report.pdf"
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True)
    return jsonify({"error": "Report not found"}), 404

# Home endpoint
@app.route('/')
def index():
    return render_template('index.html')

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
