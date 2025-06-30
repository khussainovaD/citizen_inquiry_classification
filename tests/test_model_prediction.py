import pytest
import joblib
import os

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Пути к сохранённым файлам
MODEL_PATH = "models/best_classifier_model.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"

@pytest.fixture(scope="module")
def model_and_vectorizer():
    assert os.path.exists(MODEL_PATH), "Model file not found"
    assert os.path.exists(VECTORIZER_PATH), "Vectorizer file not found"
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def test_model_prediction_is_string(model_and_vectorizer):
    model, vectorizer = model_and_vectorizer
    sample_text = "Я не получил выплату по пособию за сентябрь"
    X = vectorizer.transform([sample_text])
    prediction = model.predict(X)
    assert isinstance(prediction[0], str) or isinstance(prediction[0], int), "Prediction must return a single label"

def test_model_prediction_in_known_classes(model_and_vectorizer):
    model, vectorizer = model_and_vectorizer
    known_classes = model.classes_.tolist()
    sample_text = "Хочу узнать статус регистрации ИП"
    X = vectorizer.transform([sample_text])
    prediction = model.predict(X)
    assert prediction[0] in known_classes, f"Prediction {prediction[0]} not in known classes"

def test_model_predict_proba_shape(model_and_vectorizer):
    model, vectorizer = model_and_vectorizer
    sample_text = "Почему нет начислений по субсидии"
    X = vectorizer.transform([sample_text])
    probabilities = model.predict_proba(X)
    assert probabilities.shape[1] == len(model.classes_), "Probability output should match number of classes"
