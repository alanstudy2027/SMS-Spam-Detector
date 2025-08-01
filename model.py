import joblib
import streamlit as st
import time

model = joblib.load("alan-sms")
vec = joblib.load("alan-vec")

st.title("📩 Spam Message Detector")

user = st.text_area("Enter the SMS message")

if st.button("Check SMS"):
    if user.strip() == "":
        st.warning("Please enter a message.")
    else:
        
        prog = st.progress(0)
        for i in range(101):
            time.sleep(0.01)
            prog.progress(i)
        
        trans = vec.transform([user])
        res = model.predict(trans)

        if res[0] == 0:
            st.success("✅ This is a safe message.")
        else:
            st.error("🚨 This is a spam message.")
