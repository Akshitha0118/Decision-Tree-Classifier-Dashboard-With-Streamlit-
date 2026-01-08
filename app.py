# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# =========================================================
# 2. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Decision Tree Classifier Dashboard",
    layout="wide"
)

# =========================================================
# 3. CUSTOM CSS (DARK BLACK THEME)
# =========================================================
st.markdown("""
<style>
.stApp {
    background-color: #0b0f14;
}

/* Titles */
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    color: #ffffff;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #9ca3af;
}

/* Metric cards */
.metric-box {
    background: linear-gradient(145deg, #111827, #020617);
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.9);
    text-align: center;
    color: white;
}

/* Headers */
h2, h3 {
    color: #e5e7eb;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: #020617;
    color: white;
}

/* Horizontal line */
hr {
    border: 1px solid #1f2937;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 4. TITLE SECTION
# =========================================================
st.markdown("""
<h1 class="main-title">🌳 Decision Tree Classifier</h1>
<p class="sub-title">Social Network Ads Dataset | Streamlit ML Dashboard</p>
<hr>
""", unsafe_allow_html=True)

# =========================================================
# 5. LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv(
        r'C:\Users\ADMIN\Downloads\3rd - KNN\3rd - KNN\Social_Network_Ads.csv'
    )

dataset = load_data()

# =========================================================
# 6. DATA PREVIEW
# =========================================================
st.subheader("📊 Dataset Preview")
st.dataframe(dataset.head())

# =========================================================
# 7. FEATURES & TARGET
# =========================================================
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# =========================================================
# 8. TRAIN-TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# =========================================================
# 9. FEATURE SCALING
# =========================================================
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =========================================================
# 10. MODEL TRAINING
# =========================================================
classifier = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=0
)
classifier.fit(X_train, y_train)

# =========================================================
# 11. PREDICTIONS
# =========================================================
y_pred = classifier.predict(X_test)

# =========================================================
# 12. METRICS
# =========================================================
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

bias = classifier.score(X_train, y_train)
variance = classifier.score(X_test, y_test)

y_pred_prob = classifier.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_prob)

# =========================================================
# 13. METRICS DISPLAY
# =========================================================
st.subheader("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-box">
        <h3>Accuracy</h3>
        <h2>{accuracy:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-box">
        <h3>Bias (Train)</h3>
        <h2>{bias:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-box">
        <h3>Variance (Test)</h3>
        <h2>{variance:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-box">
        <h3>AUC Score</h3>
        <h2>{auc_score:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# 14. CONFUSION MATRIX
# =========================================================
st.subheader("🧮 Confusion Matrix")
st.write(pd.DataFrame(
    cm,
    columns=["Predicted No", "Predicted Yes"],
    index=["Actual No", "Actual Yes"]
))

# =========================================================
# 15. ROC CURVE (DARK)
# =========================================================
st.subheader("📉 ROC Curve")

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

fig, ax = plt.subplots(facecolor="#0b0f14")
ax.set_facecolor("#020617")

ax.plot(fpr, tpr, linewidth=2, label="ROC Curve")
ax.plot([0, 1], [0, 1], linestyle="--")

ax.set_xlabel("False Positive Rate", color="white")
ax.set_ylabel("True Positive Rate", color="white")
ax.set_title("Receiver Operating Characteristic", color="white")
ax.tick_params(colors="white")
ax.legend(facecolor="#020617", labelcolor="white")

st.pyplot(fig)

# =========================================================
# 16. FOOTER
# =========================================================
st.markdown("""
<hr>
<p style="text-align:center; color:#9ca3af;">
Built with ❤️ using Streamlit | Decision Tree Classification
</p>
""", unsafe_allow_html=True)
