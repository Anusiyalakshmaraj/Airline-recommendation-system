# app.py
# === Airline Recommendation System - Single File Streamlit App ===
# Requirements:
#   pip install streamlit pandas scikit-learn matplotlib joblib xgboost imbalanced-learn

import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO
import joblib
import matplotlib.pyplot as plt

# ML libs
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Optional: for handling imbalance if present
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except Exception:
    SMOTE_AVAILABLE = False

st.set_page_config(page_title="Airline Recommendation System", layout="wide")

# === Utility functions ===
def to_csv_bytes(df):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()

@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

def save_model(obj, filename="models/sentiment_model.joblib"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(obj, filename)

def load_model(filename="models/sentiment_model.joblib"):
    if os.path.exists(filename):
        return joblib.load(filename)
    return None

# Robust parsing for stops (handles 'zero', '1 stop', numeric strings, etc.)
def parse_stops_value(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            if np.isnan(x): return np.nan
            return int(x)
        except: return np.nan
    s = str(x).strip().lower()
    word_map = {'zero':0, 'non-stop':0, 'nonstop':0, 'none':0, 'one':1, 'two':2, 'three':3, 'four':4}
    for token in s.replace('-', ' ').split():
        if token.isdigit(): return int(token)
        if token in word_map: return word_map[token]
    if 'non' in s and 'stop' in s: return 0
    digits = ''.join(ch for ch in s if ch.isdigit())
    if digits:
        try: return int(digits)
        except: pass
    return np.nan

# === Data load & preprocess (shared) ===
@st.cache_data
def prepare_datasets(flight_path="Clean_Dataset.csv", sentiment_path="Indian_Domestic_Airline.csv"):
    # Load files
    df_flights = load_csv(flight_path)
    df_sent = load_csv(sentiment_path)

    # Flights preprocessing
    if not df_flights.empty:
        # Convert numeric fields
        if 'price' in df_flights.columns:
            df_flights['price'] = pd.to_numeric(df_flights['price'].astype(str).str.replace('[^0-9.]','',regex=True), errors='coerce')
        if 'duration' in df_flights.columns:
            df_flights['duration'] = pd.to_numeric(df_flights['duration'].astype(str).str.replace('[^0-9.]','',regex=True), errors='coerce')
        # Normalize stops
        if 'stops' in df_flights.columns:
            df_flights['stops'] = df_flights['stops'].apply(parse_stops_value)
        else:
            df_flights['stops'] = np.nan

        # Fill missing sensible defaults
        if 'price' in df_flights.columns:
            df_flights['price'].fillna(df_flights['price'].median(), inplace=True)
        else:
            df_flights['price'] = 0.0
        if 'duration' in df_flights.columns:
            df_flights['duration'].fillna(df_flights['duration'].median(), inplace=True)
        else:
            df_flights['duration'] = 0.0

        df_flights['stops'] = df_flights['stops'].fillna(df_flights['stops'].median() if df_flights['stops'].notna().any() else 0).astype(int)

    # Sentiment preprocessing - try to be flexible about column names
    if not df_sent.empty:
        # If there's a review/text column, standardize name to 'Review'
        text_cols = [c for c in df_sent.columns if c.lower() in ('review','text','comments','tweet','message')]
        if text_cols:
            df_sent = df_sent.rename(columns={text_cols[0]:'Review'})
        # If sentiment label exists, detect it
        label_cols = [c for c in df_sent.columns if 'sent' in c.lower() or 'label' in c.lower() or c.lower()=='target']
        if label_cols:
            df_sent = df_sent.rename(columns={label_cols[0]:'Sentiment_Label'})
        # If Sentiment_Label numeric but not encoded, keep as-is; else try to map strings to common classes
        if 'Sentiment_Label' in df_sent.columns:
            # For textual labels keep as-is; for numeric leave as-is
            pass
    return df_flights, df_sent

# === SECTION: Model builders (sentiment classifiers) ===
def build_sentiment_models():
    # Return dict of model name -> sklearn estimator (not yet in pipeline with TFIDF)
    model_lr = LogisticRegression(max_iter=500, random_state=42)
    model_nb = MultinomialNB()
    model_svm = SVC(probability=True, kernel='linear', random_state=42)
    model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model_xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')
    return {
        'Logistic Regression': model_lr,
        'Naive Bayes': model_nb,
        'SVM': model_svm,
        'Random Forest': model_rf,
        'XGBoost': model_xgb
    }

# === SECTION: Sentiment training pipeline (text OR numeric features) ===
def train_sentiment_models(df_sent, sample_limit=5000, use_text=True, oversample=True):
    """
    Train the classifiers on df_sent.
      - If use_text=True and 'Review' exists: use TF-IDF on 'Review' as X.
      - Otherwise, use numeric columns (price, duration, stops) if present.
    Returns:
      - dict of results: {model_name: {model: pipeline, accuracy, f1, report, confusion}}
      - label_encoder (fitted)
    """
    results = {}
    if df_sent.empty:
        return results, None

    # Select subset to speed training if very large
    if sample_limit and len(df_sent) > sample_limit:
        df_sent = df_sent.sample(sample_limit, random_state=42).reset_index(drop=True)

    # Determine X, y
    if use_text and 'Review' in df_sent.columns:
        X_text = df_sent['Review'].astype(str).fillna('')
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        X = vectorizer.fit_transform(X_text)
    else:
        # Numeric features fallback
        fallback_cols = [c for c in ['price','duration','stops'] if c in df_sent.columns]
        if len(fallback_cols) == 0:
            # If no features, convert review to TF-IDF automatically
            if 'Review' in df_sent.columns:
                X_text = df_sent['Review'].astype(str).fillna('')
                vectorizer = TfidfVectorizer(max_features=2000)
                X = vectorizer.fit_transform(X_text)
            else:
                # can't train
                return results, None
        else:
            X = df_sent[fallback_cols].fillna(0).values

    # Target
    if 'Sentiment_Label' not in df_sent.columns:
        st.warning("Sentiment dataset has no 'Sentiment_Label' column. Please ensure labels exist for training.")
        return results, None

    y = df_sent['Sentiment_Label'].astype(str).values  # convert to str to keep consistent categories
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    # Optionally oversample (SMOTE) if imbalance and available
    if oversample and SMOTE_AVAILABLE and isinstance(X, (np.ndarray, np.matrix)):
        try:
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X, y_enc)
            X_train_full = X_res
            y_train_full = y_res
        except Exception:
            X_train_full = X
            y_train_full = y_enc
    else:
        X_train_full = X
        y_train_full = y_enc

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full if len(set(y_train_full))>1 else None)

    # build models
    models = build_sentiment_models()
    for name, estimator in models.items():
        try:
            # If vectorized sparse matrix then estimator expects array-like or sparse - OK
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            y_pred_dec = label_encoder.inverse_transform(y_pred)
            y_test_dec = label_encoder.inverse_transform(y_test)
            acc = accuracy_score(y_test_dec, y_pred_dec)
            f1 = f1_score(y_test_dec, y_pred_dec, average='weighted')
            results[name] = {
                'model': estimator,
                'accuracy': acc,
                'f1': f1,
                'report': classification_report(y_test_dec, y_pred_dec, zero_division=0, output_dict=True),
                'confusion': confusion_matrix(y_test_dec, y_pred_dec).tolist()
            }
        except Exception as e:
            results[name] = {'error': str(e)}
    # Also return vectorizer if text used
    extra = {'vectorizer': vectorizer if 'vectorizer' in locals() else None}
    return (results, label_encoder, extra)

# === SECTION: Recommendation ranking logic ===
def composite_ranking(df_flights, weights={'price':0.5, 'duration':0.3, 'stops':0.2}, sentiment_adj=None, top_n=10):
    df = df_flights.copy()
    # Select numeric comps
    comp = pd.DataFrame()
    comp['price'] = df['price'].astype(float)
    comp['duration'] = df['duration'].astype(float)
    comp['stops'] = df['stops'].astype(float)
    # Min-max scale each
    comp_scaled = (comp - comp.min()) / (comp.max() - comp.min() + 1e-9)
    df['composite_score'] = comp_scaled['price']*weights['price'] + comp_scaled['duration']*weights['duration'] + comp_scaled['stops']*weights['stops']
    # apply sentiment_adj which should be numeric per row (lower is better)
    if sentiment_adj is not None and 'Sentiment_Label' in df.columns:
        # sentiment_adj is a dict mapping label->adjustment (e.g., {'Positive': -0.1,...})
        df['composite_score'] += df['Sentiment_Label'].map(sentiment_adj).fillna(0)
    df_sorted = df.sort_values('composite_score').reset_index(drop=True)
    return df_sorted.head(top_n)

# === SECTION: App UI & pages ===
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("", ["Home", "Recommendation", "Sentiment Analysis", "About Project"])

    # Load datasets
    df_flights, df_sent = prepare_datasets("Clean_Dataset.csv", "Indian_Domestic_Airline.csv")

    # Keep models and encoder in session_state to avoid retrain unless user requests
    if 'sentiment_models' not in st.session_state:
        st.session_state['sentiment_models'] = None
    if 'label_encoder' not in st.session_state:
        st.session_state['label_encoder'] = None
    if 'vectorizer' not in st.session_state:
        st.session_state['vectorizer'] = None
    if 'trained_results' not in st.session_state:
        st.session_state['trained_results'] = None

    # === SECTION: Home ===
    if page == "Home":
        st.title("Airline Recommendation System")
        st.markdown("""
        **Welcome!** This application recommends flights combining flight attributes
        (price, duration, stops) with passenger sentiment analysis.
        Use the sidebar to navigate to **Recommendation** and **Sentiment Analysis** pages.
        """)
        # Quick stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Flights in dataset", len(df_flights) if not df_flights.empty else 0)
        col2.metric("Sentiment rows", len(df_sent) if not df_sent.empty else 0)
        col3.metric("Models ready", "Yes" if st.session_state['trained_results'] else "No")
        st.markdown("---")
        st.subheader("Design preview")
        st.image('model.png', use_column_width=False, width=700)  # your uploaded design

    # === SECTION: Recommendation ===
    elif page == "Recommendation":
        st.title("Recommendation Page")
        st.markdown("Filter the flights, set weights, and get ranked recommendations. You can download the results as CSV.")
        if df_flights.empty:
            st.warning("Clean_Dataset.csv not found or empty. Place the file and reload.")
            return

        # Filters layout similar to your design
        with st.form("rec_form"):
            c1, c2, c3 = st.columns([1,1,1])
            src = c1.selectbox("Source", options=[''] + sorted(df_flights['source'].dropna().unique().tolist()) if 'source' in df_flights.columns else [''])
            dst = c2.selectbox("Destination", options=[''] + sorted(df_flights['destination'].dropna().unique().tolist()) if 'destination' in df_flights.columns else [''])
            airline_opt = c3.selectbox("Airline (optional)", options=[''] + sorted(df_flights['airline'].dropna().unique().tolist()) if 'airline' in df_flights.columns else [''])
            # second row
            c4, c5, c6 = st.columns([1,1,1])
            travel_class = c4.selectbox("Travel Class", options=[''] + sorted(df_flights['class'].dropna().unique().tolist()) if 'class' in df_flights.columns else [''])
            max_price = c5.number_input("Max price", min_value=0.0, value=float(df_flights['price'].median()) if 'price' in df_flights.columns else 0.0)
            max_stops = c6.slider("Max stops", min_value=0, max_value=int(df_flights['stops'].max()) if 'stops' in df_flights.columns else 3, value=2)
            top_n = st.number_input("Number of recommendations", min_value=1, max_value=50, value=10)
            st.markdown("## Weight preferences (higher value => more important)")
            w1, w2, w3 = st.columns(3)
            price_w = w1.slider("Price weight", min_value=0.0, max_value=1.0, value=0.5)
            dur_w = w2.slider("Duration weight", min_value=0.0, max_value=1.0, value=0.3)
            stops_w = w3.slider("Stops weight", min_value=0.0, max_value=1.0, value=0.2)
            use_sentiment_model = st.checkbox("Adjust ranking with sentiment model results (if trained)", value=True)
            submit = st.form_submit_button("Get Recommendations")

        if submit:
            filters = {}
            if src: filters['source'] = src
            if dst: filters['destination'] = dst
            if airline_opt: filters['airline'] = airline_opt
            if travel_class: filters['class'] = travel_class
            filters['max_price'] = max_price
            filters['stops'] = max_stops

            df_filtered = df_flights.copy()
            # apply filters
            for k,v in filters.items():
                if k in ['max_price','stops']: continue
                if k in df_filtered.columns:
                    df_filtered = df_filtered[df_filtered[k] == v]
            if 'max_price' in filters:
                df_filtered = df_filtered[df_filtered['price'] <= filters['max_price']]
            if 'stops' in filters:
                df_filtered = df_filtered[df_filtered['stops'] <= filters['stops']]

            if df_filtered.empty:
                st.warning("No results for these filters.")
            else:
                # determine sentiment adjustments using currently trained model if requested
                sentiment_adj = None
                if use_sentiment_model and st.session_state['trained_results'] and st.session_state['label_encoder']:
                    # We'll predict sentiment for flights dataset only if dataset has relevant text or Sentiment_Label exists
                    # If flights already have Sentiment_Label, use it directly
                    if 'Sentiment_Label' in df_filtered.columns:
                        # create adj map from label->adjustment value (positive reduces score)
                        unique_labels = st.session_state['label_encoder'].classes_
                        # default adj: positive -> -0.1, neutral -> 0, negative -> +0.1
                        sentiment_adj = {}
                        for lab in unique_labels:
                            lab_low = str(lab).lower()
                            if 'pos' in lab_low:
                                sentiment_adj[lab] = -0.12
                            elif 'neg' in lab_low:
                                sentiment_adj[lab] = 0.12
                            else:
                                sentiment_adj[lab] = 0.0
                    else:
                        # no sentiment label in flights; we could predict if 'Review' text exists or use default 0
                        sentiment_adj = None

                weights = {'price': price_w, 'duration': dur_w, 'stops': stops_w}
                df_ranked = composite_ranking(df_filtered, weights=weights, sentiment_adj=sentiment_adj, top_n=int(top_n))

                st.success(f"Top {len(df_ranked)} recommendations")
                st.dataframe(df_ranked.reset_index(drop=True))

                # Download button
                st.download_button("Download recommendations (CSV)", data=to_csv_bytes(df_ranked), file_name="recommendations.csv", mime="text/csv")

    # === SECTION: Sentiment Analysis ===
    elif page == "Sentiment Analysis":
        st.title("Sentiment Analysis / Model Comparison")
        st.markdown("This page trains sentiment classifiers on `Indian_Domestic_Airline.csv` and compares them. By default, dataset distribution is shown.")

        if df_sent.empty:
            st.warning("Indian_Domestic_Airline.csv not found or empty. Place the file and reload.")
            return

        # Show first rows
        st.subheader("Dataset preview")
        st.dataframe(df_sent.head(5))

        # Default distribution (if Sentiment_Label exists)
        if 'Sentiment_Label' in df_sent.columns:
            st.subheader("Sentiment distribution (dataset)")
            dist = df_sent['Sentiment_Label'].value_counts().sort_index()
            fig, ax = plt.subplots()
            bars = ax.bar(dist.index.astype(str), dist.values)
            ax.set_ylabel("Count")
            ax.set_title("Sentiment label counts in dataset")
            # annot above bars
            for rect in bars:
                height = rect.get_height()
                ax.annotate(f'{int(height)}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0,3), textcoords="offset points", ha='center', va='bottom')
            st.pyplot(fig)
        else:
            st.info("Dataset has no 'Sentiment_Label' column; default distribution can't be shown.")

        st.markdown("---")
        st.subheader("Train models (use text reviews if 'Review' column exists; otherwise numeric features)")

        use_text = st.checkbox("Use text feature (TF-IDF on 'Review') if available", value=True)
        sample_limit = st.slider("Sample max rows for training (helps speed)", min_value=500, max_value=20000, value=5000, step=500)
        oversample = st.checkbox("Use SMOTE oversampling (if imbalanced and available)", value=False)

        # buttons to train single model or all
        colA, colB = st.columns(2)
        if colA.button("Train & Evaluate ALL models"):
            with st.spinner("Training all models..."):
                res, le, extra = train_sentiment_models(df_sent, sample_limit=sample_limit, use_text=use_text, oversample=oversample)
                st.session_state['trained_results'] = res
                st.session_state['label_encoder'] = le
                st.session_state['vectorizer'] = extra.get('vectorizer', None)
            st.success("Training completed.")

        # Train individual model (choose)
        models_available = list(build_sentiment_models().keys())
        sel_model = colB.selectbox("Train single model", options=models_available)
        if colB.button("Train selected model"):
            with st.spinner(f"Training {sel_model} ..."):
                all_res, le, extra = train_sentiment_models(df_sent, sample_limit=sample_limit, use_text=use_text, oversample=oversample)
                # Keep only selected model result but still store others
                st.session_state['trained_results'] = all_res
                st.session_state['label_encoder'] = le
                st.session_state['vectorizer'] = extra.get('vectorizer', None)
            st.success(f"Training completed. {sel_model} ready.")

        # If results exist, show comparison chart and details
        if st.session_state['trained_results']:
            results = st.session_state['trained_results']
            # Build comparison DataFrame
            comp_rows = []
            for name, info in results.items():
                if 'error' in info:
                    comp_rows.append({'Model': name, 'Accuracy': np.nan, 'F1': np.nan})
                else:
                    comp_rows.append({'Model': name, 'Accuracy': info.get('accuracy', np.nan), 'F1': info.get('f1', np.nan)})
            comp_df = pd.DataFrame(comp_rows).set_index('Model').sort_values('Accuracy', ascending=False)
            st.subheader("Model comparison")
            st.dataframe(comp_df.round(4))

            # bar chart with values on top
            fig2, ax2 = plt.subplots(figsize=(8, max(4, len(comp_df)*0.6)))
            bars = ax2.bar(comp_df.index.astype(str), comp_df['Accuracy'].fillna(0).values)
            ax2.set_ylabel("Accuracy")
            ax2.set_ylim(0,1)
            ax2.set_title("Model Accuracy Comparison")
            for rect, val in zip(bars, comp_df['Accuracy'].fillna(0).values):
                ax2.annotate(f"{val:.3f}", xy=(rect.get_x() + rect.get_width() / 2, val), xytext=(0,3), textcoords="offset points", ha='center', va='bottom')
            plt.xticks(rotation=30, ha='right')
            st.pyplot(fig2)

            # allow user to click a model to see detailed report
            sel = st.selectbox("Select model to view detailed classification report", options=comp_df.index.tolist())
            info = results[sel]
            if 'error' in info:
                st.error(f"Training error for {sel}: {info['error']}")
            else:
                st.subheader(f"{sel} - metrics")
                st.write(f"Accuracy: {info['accuracy']:.4f} | F1-weighted: {info['f1']:.4f}")
                rpt = pd.DataFrame(info['report']).transpose().round(3)
                st.dataframe(rpt)
                st.write("Confusion matrix:")
                st.table(info['confusion'])

            # Save best model option
            if st.button("Save best model to disk"):
                best_name = comp_df['Accuracy'].idxmax()
                best_info = results[best_name]
                save_model(best_info['model'], filename=f"models/{best_name.replace(' ','_')}_sentiment.joblib")
                st.success(f"Saved {best_name} to models/{best_name.replace(' ','_')}_sentiment.joblib")

    # === SECTION: About Project ===
    elif page == "About Project":
        st.title("About Project")
        st.markdown("""
        **Airline Recommendation System**  
        - **Objective:** Recommend flights combining price, duration, stops, and passenger sentiment to improve passenger satisfaction.  
        - **Datasets used:** `Clean_Dataset.csv` (flights) and `Indian_Domestic_Airline.csv` (sentiment).  
        - **Models implemented:** Logistic Regression, Naive Bayes, SVM, Random Forest, XGBoost, and Voting/Ensemble patterns where needed.  
        - **Evaluation metrics:** Accuracy, F1-Score, Classification Report, Confusion Matrix.  
        - **Workflow:** Data Cleaning -> Feature Engineering -> Model Training -> Recommendation Ranking -> Evaluation -> Deployment.
        """)
        st.markdown("**Future scope:** integrate explicit user profile, real-time feedback loop, deploy with authentication and database backend.")

if __name__ == "__main__":
    main()
