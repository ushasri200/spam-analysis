# spam_app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# 1. Load and Prepare Dataset
# -----------------------------
st.title("📧 Spam Mail Prediction App")
st.write("This app predicts whether a message is Spam or Not Spam.")

@st.cache_data
def load_data():
    # Load dataset (downloaded from UCI: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
    df = pd.read_csv("spam.csv", encoding="latin-1")[["v1","v2"]]
    df.columns = ["label", "message"]
    df["label_num"] = df.label.map({"ham":0, "spam":1})
    return df

data = load_data()

# -----------------------------
# 2. Train Model
# -----------------------------
@st.cache_resource
def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label_num"], test_size=0.2, random_state=42
    )
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)
    
    model = MultinomialNB()
    model.fit(X_train_counts, y_train)
    
    return model, vectorizer

model, vectorizer = train_model(data)

# -----------------------------
# 3. User Input
# -----------------------------
user_input = st.text_area("Enter your email/message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message to predict!")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        prediction_proba = model.predict_proba(input_vector)[0]

        # Display results
        if prediction == 1:
            st.error("⚠️ This message is likely **SPAM**!")
        else:
            st.success("✅ This message is **NOT SPAM**.")

        st.write("Prediction Probabilities:")
        st.write(f"Not Spam: {prediction_proba[0]:.2f}, Spam: {prediction_proba[1]:.2f}")

# -----------------------------
# Optional: Show Dataset Sample
# -----------------------------
if st.checkbox("Show sample data"):
    st.write(data.sample(5))