import streamlit as st
import joblib
import os

# Загрузка модели и векторизатора
model_path = os.path.join("models", "best_classifier_model.joblib")
vectorizer_path = os.path.join("models", "tfidf_vectorizer.joblib")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Интерфейс
st.set_page_config(page_title="Inquiries classification", page_icon="📨")
st.title("📨 Classification of citizens' inquiries")
st.markdown("""
Enter the message text, and the model will tell you which category it belongs to.
""")

# Ввод текста
user_input = st.text_area("Text of the inquiry", height=200)

# Предсказание
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter the text.")
    else:
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)[0]
        st.success(f"Predicted category: **{prediction}**")
