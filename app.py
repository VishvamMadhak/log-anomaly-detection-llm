import streamlit as st
import torch
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer
import torch.nn as nn
from transformers import BertForSequenceClassification
import os
import json

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model class (same as in your training code)
class LogClassifier:
    def __init__(self, model_name, num_labels=2):
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(device)
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.to(device)
        print(f"Model loaded from {path}")

# Parse a log file
def parse_bgl_logs(file_content):
    logs_data = []
    
    pattern = re.compile(
        r"[-\w]*\s*(\d+)\s+(\d{4}\.\d{2}\.\d{2})\s+([\w\-\:]+)\s+(\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+)\s+([\w\-\:]+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(.*)"
    )
    
    for line in file_content.split('\n'):
        line = line.strip()
        match = pattern.match(line)
        if match:
            epoch, date, location, timestamp, component, category, source, severity, message = match.groups()
            logs_data.append({
                'epoch': epoch,
                'date': date,
                'location': location,
                'timestamp': timestamp,
                'component': component,
                'category': category,
                'source': source,
                'severity': severity,
                'message': message
            })
    
    return logs_data

# Function to predict anomaly
def predict_anomaly(log_entry, classifier, tokenizer):
    # Tokenize the message
    message = log_entry['message']
    encoding = tokenizer(
        message,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = classifier.model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs.logits, dim=1)
    
    # Get probability
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    anomaly_prob = probs[0][1].item()
    
    return {
        'is_anomaly': prediction.item(),
        'anomaly_probability': anomaly_prob,
        'prediction': 'ANOMALY' if prediction.item() == 1 else 'NORMAL'
    }

# Save uploaded files
def save_uploaded_file(uploaded_file, save_dir='temp'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

# Load hyperparameters
def load_hyperparams(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Streamlit app
def main():
    st.set_page_config(page_title="Log Anomaly Detector", layout="wide")
    
    st.title("Log Anomaly Detection Dashboard")
    st.write("Upload your trained TinyBERT model, hyperparameters, and log file for analysis")
    
    # File uploaders
    col1, col2, col3 = st.columns(3)
    with col1:
        model_file = st.file_uploader("Upload Model (.pt file)", type=['pt'])
    with col2:
        hyperparams_file = st.file_uploader("Upload Hyperparameters (.json)", type=['json'])
    with col3:
        log_file = st.file_uploader("Upload Log File (.log)", type=['log', 'txt'])
    
    # Initialize session state
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'logs_df' not in st.session_state:
        st.session_state.logs_df = None
    
    # Load model if uploaded
    if model_file and hyperparams_file and not st.session_state.model_loaded:
        with st.spinner("Loading model..."):
            # Save files temporarily
            model_path = save_uploaded_file(model_file)
            hyperparams_path = save_uploaded_file(hyperparams_file)
            
            # Load hyperparameters (not used directly now but good to have)
            hyperparams = load_hyperparams(hyperparams_path)
            
            # Initialize tokenizer
            tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
            
            # Load model
            classifier = LogClassifier('huawei-noah/TinyBERT_General_4L_312D')
            classifier.load(model_path)
            classifier.model.eval()
            
            st.session_state.classifier = classifier
            st.session_state.tokenizer = tokenizer
            st.session_state.hyperparams = hyperparams
            st.session_state.model_loaded = True
            
            st.success("Model loaded successfully!")
    
    # Process log file if model is loaded
    if st.session_state.model_loaded and log_file:
        # Read log file
        log_content = log_file.getvalue().decode('utf-8')
        
        # Parse logs
        with st.spinner("Parsing log file..."):
            logs_data = parse_bgl_logs(log_content)
            logs_df = pd.DataFrame(logs_data)
            st.session_state.logs_df = logs_df
            
            # Make predictions
            predictions = []
            with st.spinner("Analyzing logs..."):
                progress_bar = st.progress(0)
                for i, log in enumerate(logs_data):
                    prediction = predict_anomaly(log, st.session_state.classifier, st.session_state.tokenizer)
                    log.update(prediction)
                    predictions.append(log)
                    progress_bar.progress((i + 1) / len(logs_data))
            
            # Store predictions
            predictions_df = pd.DataFrame(predictions)
            st.session_state.predictions = predictions_df
    
    # Display results if predictions are available
    if st.session_state.predictions is not None:
        st.header("Analysis Results")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        total_logs = len(st.session_state.predictions)
        anomaly_count = st.session_state.predictions['is_anomaly'].sum()
        normal_count = total_logs - anomaly_count
        
        with col1:
            st.metric("Total Logs", total_logs)
        with col2:
            st.metric("Anomalies Detected", anomaly_count)
        with col3:
            st.metric("Normal Logs", normal_count)
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["All Logs", "Anomalies Only", "Visualizations"])
        
        with tab1:
            st.dataframe(st.session_state.predictions[['epoch', 'component', 'category', 'severity', 'message', 'prediction', 'anomaly_probability']])
        
        with tab2:
            anomalies = st.session_state.predictions[st.session_state.predictions['is_anomaly'] == 1]
            if len(anomalies) > 0:
                st.dataframe(anomalies[['epoch', 'component', 'category', 'severity', 'message', 'anomaly_probability']])
            else:
                st.info("No anomalies detected in the log file")
        
        with tab3:
            # Visualization 1: Component-wise anomaly distribution
            st.subheader("Component-wise Anomaly Distribution")
            component_counts = st.session_state.predictions.groupby(['component', 'prediction']).size().reset_index(name='count')
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='component', y='count', hue='prediction', data=component_counts, ax=ax)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Visualization 2: Anomaly Probability Distribution
            st.subheader("Anomaly Probability Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(st.session_state.predictions['anomaly_probability'], bins=20, kde=True, ax=ax)
            plt.xlabel('Anomaly Probability')
            plt.ylabel('Count')
            st.pyplot(fig)
            
            # Visualization 3: Severity Distribution
            st.subheader("Severity vs Prediction")
            severity_counts = st.session_state.predictions.groupby(['severity', 'prediction']).size().reset_index(name='count')
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='severity', y='count', hue='prediction', data=severity_counts, ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Export results
            csv = st.session_state.predictions.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="log_analysis_results.csv",
                mime="text/csv"
            )

# Run the app
if __name__ == "__main__":
    main()