import streamlit as st
import pandas as pd
import requests

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="📧",
    layout="centered"
)

# =========================
# HEADER
# =========================
st.markdown("""
# 📧 AI Spam Detection System
### Detect spam messages using Machine Learning
""")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("📊 Project Info")

    st.write("""
    **AI Spam Detection System**

    This system detects spam messages using:

    - TF-IDF Vectorization
    - Naive Bayes Machine Learning
    - FastAPI Backend
    - Streamlit Web Interface
    """)

# =========================
# DESCRIPTION
# =========================
st.write("Detect whether a message is **Spam or Not Spam** using Machine Learning.")
st.divider()

# =========================
# INPUT
# =========================
message = st.text_area("Enter a message to analyze")

# =========================
# HISTORY STORAGE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# BUTTON ACTION
# =========================
if st.button("Analyze Message 🔍"):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message first.")

    else:
        try:
            # LOCAL FASTAPI URL (for testing on your laptop)
            # Later replace with Render URL
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"text": message},
                timeout=10
            )

            result_data = response.json()

            prediction = result_data["prediction"]
            spam_conf = result_data["spam_probability"]
            ham_conf = 1 - spam_conf
            important_words = result_data.get("important_words", [])

            st.subheader("Prediction Result")

            if prediction == "Spam":
                result = "Spam"
                st.error("🚨 This message is SPAM")
            else:
                result = "Not Spam"
                st.success("✅ This message is NOT SPAM")

            st.divider()

            # =========================
            # CONFIDENCE SCORE
            # =========================
            st.subheader("Confidence Score")

            st.metric("Spam Probability", f"{spam_conf*100:.2f}%")
            st.metric("Not Spam Probability", f"{ham_conf*100:.2f}%")

            st.progress(float(spam_conf))

            st.divider()

            # =========================
            # AI EXPLANATION
            # =========================
            st.subheader("🤖 Why did the model predict this?")

            if prediction == "Spam":
                st.info(
                    "The model detected suspicious words or patterns commonly found in spam messages."
                )
            else:
                st.info(
                    "The message appears normal and does not strongly match common spam patterns."
                )

            st.divider()

            # =========================
            # IMPORTANT WORDS
            # =========================
            st.subheader("🔍 Important Words Detected")

            if important_words:
                st.write(", ".join(important_words[:10]))
            else:
                st.write("No major keywords detected.")

            # =========================
            # SAVE HISTORY
            # =========================
            st.session_state.history.append({
                "Message": message,
                "Prediction": result,
                "Spam Probability": f"{spam_conf*100:.2f}%"
            })

        except Exception as e:
            st.error("⚠️ FastAPI backend is not running or not reachable.")
            st.info("Please start FastAPI first using: uvicorn api:app --reload")

# =========================
# SHOW HISTORY
# =========================
if len(st.session_state.history) > 0:
    st.divider()
    st.subheader("📜 Prediction History")

    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)