# app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from fpdf import FPDF

st.set_page_config(layout="wide")
st.title("📊 Salary Classification Web App")

uploaded_file = st.file_uploader("Upload your dataset (.xlsx)", type=["xlsx"])

model_option = st.selectbox("Select Model", ["RandomForest", "SVM", "XGBoost"])

def generate_pdf(images, accuracy, report):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Salary Prediction Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Accuracy: {accuracy:.2%}", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=report)

    for img_title, img_buf in images.items():
        pdf.add_page()
        pdf.image(img_buf, w=180)

    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("📁 Uploaded Data Preview:", df.head())

    df.dropna(inplace=True)
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    target_options = ['income', 'salary', 'Income', 'Salary', 'class', 'target']
    target_col = next((col for col in target_options if col in df.columns), None)

    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_option == "RandomForest":
            model = RandomForestClassifier(random_state=42)
        elif model_option == "SVM":
            model = SVC(probability=True)
        elif model_option == "XGBoost":
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        fi = None
        if hasattr(model, 'feature_importances_'):
            fi = model.feature_importances_

        st.metric(label="✅ Accuracy", value=f"{acc:.2%}")

        # Prepare images for PDF
        image_buffers = {}

        # Classification Report
        report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().iloc[:-1, :-1]
        fig1, ax1 = plt.subplots()
        sns.heatmap(report_df, annot=True, cmap='Blues', fmt='.2f', ax=ax1)
        ax1.set_title("Classification Report")
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png')
        buf1.seek(0)
        st.pyplot(fig1)
        image_buffers["Classification Report"] = buf1

        # Confusion Matrix
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax2)
        ax2.set_title("Confusion Matrix")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png')
        buf2.seek(0)
        st.pyplot(fig2)
        image_buffers["Confusion Matrix"] = buf2

        # Feature Importance
        if fi is not None:
            fi_df = pd.DataFrame({'Feature': X.columns, 'Importance': fi}).sort_values(by='Importance')
            fig3, ax3 = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=fi_df, ax=ax3)
            ax3.set_title("Feature Importance")
            buf3 = io.BytesIO()
            fig3.savefig(buf3, format='png')
            buf3.seek(0)
            st.pyplot(fig3)
            image_buffers["Feature Importance"] = buf3

        # Actual vs Predicted
        result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        fig4, ax4 = plt.subplots()
        sns.countplot(x='Actual', data=result_df, color='blue', alpha=0.6, ax=ax4)
        sns.countplot(x='Predicted', data=result_df, color='red', alpha=0.4, ax=ax4)
        ax4.legend(["Actual", "Predicted"])
        ax4.set_title("Actual vs Predicted")
        buf4 = io.BytesIO()
        fig4.savefig(buf4, format='png')
        buf4.seek(0)
        st.pyplot(fig4)
        image_buffers["Actual vs Predicted"] = buf4

        # Display predictions
        st.markdown("### 📋 Sample Predictions")
        st.dataframe(result_df.head(20))

        # Downloadable PDF report
        st.markdown("### 📄 Download Report")
        pdf_data = generate_pdf(image_buffers, acc, report)
        st.download_button(label="📥 Download PDF Report", data=pdf_data, file_name="salary_prediction_report.pdf", mime="application/pdf")

    else:
        st.error("❌ Target column not found. Please rename it to one of: income, salary, Income, Salary, class, target")
