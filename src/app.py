import os
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from src.feats import UrlStatFeatures

DATA_URL = "https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv"
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "svm_url_spam.joblib")

def load_data():
    df = pd.read_csv(DATA_URL)
    if "url" not in df.columns:
        raise ValueError("El dataset debe tener columna 'url'")
    if "label" in df.columns:
        y = df["label"]
    elif "is_spam" in df.columns:
        y = df["is_spam"]
    elif "target" in df.columns:
        y = df["target"]
    else:
        raise ValueError("No encuentro la columna de etiqueta (label/is_spam/target)")
    X = pd.DataFrame({"url": df["url"].astype(str)})
    return X, y.astype(int)

def build_pipeline(tfidf_min_df=3, tfidf_ngram=(1,3), C=1.0):
    feats_block = ("stats", UrlStatFeatures(), "url")
    tfidf_block = ("tfidf", TfidfVectorizer(
        min_df=tfidf_min_df,
        ngram_range=tfidf_ngram,
        token_pattern=r"[^/?:&.=]+"
    ), "url")

    pre = ColumnTransformer(
        transformers=[feats_block, tfidf_block],
        remainder="drop",
        verbose_feature_names_out=False
    )
    clf = LinearSVC(C=C, max_iter=5000)
    pipe = Pipeline([
        ("features", pre),
        ("clf", clf)
    ])
    return pipe

def train_and_save(all_steps=True):
    X, y = load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    base = build_pipeline()
    base.fit(Xtr, ytr)
    ypred = base.predict(Xte)
    acc = accuracy_score(yte, ypred)
    print(f"Baseline accuracy: {acc:.4f}")

    if all_steps:
        param_grid = {
            "features__tfidf__min_df": [1, 2, 3],
            "features__tfidf__ngram_range": [(1,2), (1,3), (2,5)],
            "clf__C": [0.5, 1.0, 2.0]
        }
        gs = GridSearchCV(base, param_grid, cv=5, n_jobs=-1, verbose=0)
        gs.fit(Xtr, ytr)
        print(f"Best params: {gs.best_params_}")
        best = gs.best_estimator_
    else:
        best = base

    ypred = best.predict(Xte)
    acc = accuracy_score(yte, ypred)
    print(f"Eval accuracy: {acc:.4f}")
    print(classification_report(yte, ypred, digits=2))

    joblib.dump(best, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Entrenar + buscar hiperparámetros + guardar")
    parser.add_argument("--fast", action="store_true", help="Entrena sin grid-search")
    args = parser.parse_args()

    if args.fast:
        train_and_save(all_steps=False)
    else:
        train_and_save(all_steps=True)

if __name__ == "__main__":
    main()
