# Import necessary libraries
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import time
import streamlit as st  # For real-time dashboard

# Define the Log Analyzer class
class RealTimeLogAnalyzer:
    def __init__(self):
        # Load pre-trained tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=4  # INFO, WARNING, ERROR, CRITICAL
        )
        self.severity_labels = ['INFO', 'WARNING', 'ERROR', 'CRITICAL']

    def analyze_log(self, log_message):
        """Analyze a log message and predict its severity"""
        # Tokenize the log message
        inputs = self.tokenizer(log_message, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get predicted severity and confidence
        predicted_label = self.severity_labels[predictions.argmax().item()]
        confidence = predictions.max().item()
        
        return predicted_label, confidence

# Simulate real-time log generation
def simulate_real_time_logs(dataset):
    """Simulate real-time log generation from a dataset"""
    for index, row in dataset.iterrows():
        log_message = row['log_message']
        yield log_message
        time.sleep(2)  # Simulate a 2-second delay between logs

# Streamlit Dashboard for Real-Time Monitoring
def real_time_dashboard(analyzer, dataset):
    """Real-time dashboard for log monitoring"""
    st.title("Real-Time Log Monitoring Using LLMs")
    st.write("This dashboard simulates real-time log analysis using a pre-trained transformer model.")

    # Display logs and analysis in real-time
    st.header("Live Log Analysis")
    log_placeholder = st.empty()  # Placeholder to display logs
    analysis_placeholder = st.empty()  # Placeholder to display analysis

    for log_message in simulate_real_time_logs(dataset):
        # Analyze the log message
        severity, confidence = analyzer.analyze_log(log_message)
        
        # Display the log and analysis
        log_placeholder.markdown(f"**Log Message:** `{log_message}`")
        analysis_placeholder.markdown(f"""
            **Predicted Severity:** `{severity}`  
            **Confidence:** `{confidence:.2f}`
        """)
        
        # Add a small delay to simulate real-time processing
        time.sleep(2)

# Main function
def main():
    # Load your dataset
    # Replace 'log_analysis_dataset.csv' with your dataset file
    dataset = pd.read_csv('log_analysis_dataset.csv')

    # Ensure the dataset has a 'log_message' column
    if 'log_message' not in dataset.columns:
        st.error("The dataset must contain a 'log_message' column.")
        return

    # Initialize the log analyzer
    analyzer = RealTimeLogAnalyzer()

    # Run the real-time dashboard
    real_time_dashboard(analyzer, dataset)

# Run the program
if __name__ == "__main__":
    main()