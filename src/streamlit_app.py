import os, sys, joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feats import UrlStatFeatures 

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "svm_url_spam.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def predict_url(model, url: str):
    X = pd.DataFrame({"url": [url]})
    pred = model.predict(X)[0]
    score = None
    if hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[0])
    return int(pred), score

def main():
    st.title("Detector de URLs Spam")
    st.write("Clasifica una URL como spam o ham usando el modelo SVM entrenado.")
    url = st.text_input("Pega una URL para evaluar:", value="")
    if st.button("Predecir") and url.strip():
        model = load_model()
        pred, score = predict_url(model, url.strip())
        st.success(f"Predicción: {'spam' if pred==1 else 'ham'}")
        if score is not None:
            st.caption(f"score: {score:.3f}")

if __name__ == "__main__":
    main()
