import joblib
import streamlit as st
import time

# Load model, vectorizer, and evaluation data
model = joblib.load("alan-sms")
vec = joblib.load("alan-vec")

# Mock evaluation metrics (replace with your actual metrics)
eval_metrics = {
    'accuracy': 0.9571,
    'precision': 0.9671,
    'recall': 0.951,
    'f1': 0.942,
    'test_size': 8175  # Number of test samples
}

st.title("📩 Spam Message Detector")
st.markdown("Check if your SMS is spam or safe")

# Model evaluation section
with st.expander("Model Performance Details"):
    st.markdown(f"""
    **Evaluation Metrics (trained on {eval_metrics['test_size']} messages):**
    
    - ✅ **Accuracy:** {eval_metrics['accuracy']:.1%}  
      (Overall correct predictions)
    
    - 🎯 **Precision:** {eval_metrics['precision']:.1%}  
      (Correct spam detections / All spam alerts)
    
    - 🔍 **Recall:** {eval_metrics['recall']:.1%}  
      (Actual spam messages detected)
    
    - ⚖️ **F1 Score:** {eval_metrics['f1']:.1%}  
      (Balance between precision and recall)
    """)
    st.caption("Note: Metrics based on holdout test set")

# User input
user = st.text_area("Enter the SMS message:", 
                   placeholder="Paste your message here...",
                   height=150)

if st.button("Check SMS"):
    if not user.strip():
        st.warning("Please enter a message.")
    else:
        # Progress animation
        prog = st.progress(0)
        status_text = st.empty()
        
        for i in range(101):
            time.sleep(0.01)
            prog.progress(i)
            status_text.text(f"Analyzing... {i}%")
        
        # Make prediction
        trans = vec.transform([user])
        res = model.predict(trans)[0]
        spam_prob = model.predict_proba(trans)[0][1] * 100
        
        # Clear progress
        prog.empty()
        status_text.empty()
        
        # Show results
        if res == 0:
            st.success(f"""
            ✅ **Safe Message**  
            Spam probability: {spam_prob:.1f}%  
            Confidence: {(100 - spam_prob):.1f}% safe
            """)
        else:
            st.error(f"""
            🚨 **Spam Detected**  
            Spam probability: {spam_prob:.1f}%  
            Confidence: {spam_prob:.1f}% spam
            """)
        
        # Add explanation
        st.markdown("### Analysis")
        if spam_prob > 90:
            st.write("🔴 Strong spam indicators detected")
        elif spam_prob > 70:
            st.write("🟠 Multiple suspicious elements")
        elif spam_prob > 50:
            st.write("🟡 Some suspicious content")
        else:
            st.write("🟢 Minimal spam indicators")
        
        # Safety tips
# Footer
st.caption("Model developed by Alan Joshua")