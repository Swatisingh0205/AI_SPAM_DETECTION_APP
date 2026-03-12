import streamlit as st
import pickle
import time

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page settings
st.set_page_config(
    page_title="Spam Detection AI",
    page_icon="📩",
    layout="centered"
)

# Title
st.title("📩 AI Spam Message Detector")

st.write(
"""
This AI model analyzes messages and predicts whether they are **Spam** or **Not Spam**.
"""
)

# Message input
message = st.text_area("Enter your message here")

# Analyze button
if st.button("Analyze Message 🔍"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    
    else:
        with st.spinner("Analyzing message..."):
            time.sleep(1)

            data = vectorizer.transform([message])
            prediction = model.predict(data)[0]
            probability = model.predict_proba(data)[0][1]

        if prediction == 1:
            st.error("🚨 This message is **SPAM**")
        else:
            st.success("✅ This message is **NOT Spam**")

        st.info(f"Spam Probability: **{round(probability*100,2)}%**")

# Sidebar
st.sidebar.title("About Project")

st.sidebar.write("""
**Spam Detection using Machine Learning**

Model Used:
- Multinomial Naive Bayes

Tech Stack:
- Python
- Scikit-learn
- TF-IDF
- Streamlit
                 


""")