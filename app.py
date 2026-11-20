import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="HeartCare AI | KNN", page_icon="🫀", layout="wide")

from pathlib import Path
import base64

PROFILE_PATH = Path(__file__).parent / "assets" / "profile.jpg"
try:
    PROFILE_B64 = base64.b64encode(PROFILE_PATH.read_bytes()).decode("utf-8")
except Exception:
    PROFILE_B64 = ""

FEATURES = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]
TARGET = "HeartDisease"

st.markdown("""\n<style>
.stApp {background:linear-gradient(135deg,#07111f,#0b1d31 55%,#102941);color:#e8f3ff}
header[data-testid="stHeader"]{background:rgba(7,17,31,.85)!important}
header[data-testid="stHeader"] *{color:#f5f9ff!important;fill:#f5f9ff!important}
[data-testid="stSidebar"]{background:#081626;border-right:1px solid #183652}
[data-testid="stMetric"]{background:#112840;border:1px solid #1e4667;padding:14px;border-radius:15px}
[data-testid="stMetricLabel"]{color:#9dbbd4!important}[data-testid="stMetricValue"]{color:#f1f8ff!important}
.hero{background:linear-gradient(120deg,#103653,#0d617c);border:1px solid #226184;padding:1.3rem 1.5rem;border-radius:20px;margin-bottom:1rem}
.hero h1{margin:0;color:#effaff;font-size:2.2rem}.hero p{color:#b9e5f7;margin:.35rem 0 0}
.stButton>button, [data-testid="stDownloadButton"]>button{border-radius:12px;border:1px solid #45c5f4;background:#087fa8;color:#ffffff!important;font-weight:700}
.stButton>button:hover, [data-testid="stDownloadButton"]>button:hover{background:#0e9ac8;border-color:#6bd8ff;color:#ffffff!important}
.stButton>button *, [data-testid="stDownloadButton"]>button *{color:#ffffff!important}
div[data-testid="stTabs"] button, div[data-testid="stTabs"] button *{color:#ffffff!important}
/* Improve contrast: bright text on dark surfaces, dark text on light surfaces */
.stApp, .stApp p, .stApp label, .stApp span, .stApp li, .stApp small, .stApp div, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {color:#f5f9ff;}
[data-testid="stSidebar"] *, [data-testid="stWidgetLabel"] *, [data-testid="stMarkdownContainer"] * {color:#f5f9ff!important}
/* Inline code pills */
.stApp code, [data-testid="stMarkdownContainer"] code, code{background-color:#13314d!important;color:#7dd3fc!important;border:1px solid #245475!important;border-radius:6px!important;padding:2px 6px!important}
/* File uploader dropzone & uploaded file item */
section[data-testid="stFileUploadDropzone"], [data-testid="stFileUploadDropzone"],
section[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploader"] section{background-color:#0f263d!important;border:1px dashed #45c5f4!important;border-radius:12px!important}
section[data-testid="stFileUploadDropzone"] *, [data-testid="stFileUploadDropzone"] *,
section[data-testid="stFileUploaderDropzone"] *, [data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploader"] section *, [data-testid="stFileUploaderFile"] *{color:#f5f9ff!important;fill:#f5f9ff!important}
section[data-testid="stFileUploadDropzone"] button, [data-testid="stFileUploadDropzone"] button,
section[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] section button{background-color:#087fa8!important;border:1px solid #45c5f4!important;color:#ffffff!important}
/* Alert boxes (st.info, st.warning, st.error, st.success) */
div[role="alert"], div[role="alert"] *,
[data-testid="stAlert"], [data-testid="stAlert"] *,
[data-testid="stAlertContainer"], [data-testid="stAlertContainer"] *,
[data-testid="stNotification"], [data-testid="stNotification"] *,
div[role="alert"] [data-testid="stMarkdownContainer"] *,
[data-testid="stAlertContainer"] [data-testid="stMarkdownContainer"] *{color:#0f172a!important}
/* Tables & Dataframes */
[data-testid="stDataFrame"], [data-testid="stTable"]{background:#ffffff!important;border-radius:10px}
[data-testid="stTable"] *{color:#0f172a!important}
/* Form inputs & dropdowns */
input, textarea, select, [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="base-input"]{background:#ffffff!important;color:#0f172a!important}
[data-baseweb="select"] *, [data-baseweb="input"] *, [data-baseweb="base-input"] *{color:#0f172a!important}
[data-baseweb="popover"], [data-baseweb="popover"] > div, [role="listbox"], [role="option"]{background:#ffffff!important;color:#0f172a!important}
[data-baseweb="popover"] *, [role="listbox"] *, [role="option"] *{color:#0f172a!important}
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"]{background:#e2e8f0!important;color:#0f172a!important}
[data-testid="stNumberInputStepDown"] *, [data-testid="stNumberInputStepUp"] *{color:#0f172a!important;fill:#0f172a!important}
.profile-wrap{display:flex;align-items:center;gap:14px;background:linear-gradient(120deg,#0e263c,#123d59);border:1px solid #245475;border-radius:18px;padding:12px 16px;margin-bottom:14px}
.profile-photo{width:62px;height:62px;border-radius:50%;object-fit:cover;border:2px solid #58c9f5;flex-shrink:0}
.profile-info{display:flex;flex-direction:column;gap:2px}
.profile-name{font-size:1.05rem;font-weight:750;color:#fff!important}
.profile-major{font-size:.88rem;color:#d9efff!important}
.profile-id{font-size:.82rem;color:#c2d9ee!important}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_csv(file):
    return pd.read_csv(file)


def clean_data(df):
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        return None, missing
    clean = df[FEATURES + [TARGET]].copy()
    for c in clean.columns:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")
    clean = clean.dropna().reset_index(drop=True)
    if clean.empty:
        return clean, ["ไม่พบข้อมูลที่สมบูรณ์หลังจัดการค่าว่าง"]
    clean[TARGET] = clean[TARGET].astype(int)
    if clean[TARGET].nunique() < 2:
        return clean, ["คอลัมน์ HeartDisease ต้องมีอย่างน้อย 2 กลุ่ม"]
    return clean, []

if PROFILE_B64:
    profile_html = f'<div class="profile-wrap"><img class="profile-photo" src="data:image/jpeg;base64,{PROFILE_B64}"/><div class="profile-info"><div class="profile-name">ปวเรศ ภักดีเจริญ</div><div class="profile-major">Information Technology</div><div class="profile-id">674230005</div></div></div>'
    st.markdown(profile_html, unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🫀 HeartCare AI</h1><p>Heart disease classification • Machine Learning with K-Nearest Neighbors (KNN)</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ ตั้งค่าโมเดล")
    uploaded = st.file_uploader("อัปโหลด Heart Dataset (.csv)", type=["csv"])
    k_value = st.slider("จำนวนเพื่อนบ้าน K", min_value=1, max_value=25, value=5)
    test_percent = st.slider("ข้อมูลทดสอบ (%)", min_value=10, max_value=40, value=20, step=5)
    st.caption("KNN ใช้ StandardScaler เพื่อปรับสเกลก่อนคำนวณระยะห่าง")
    st.markdown("---")
    st.markdown("**คอลัมน์ที่ต้องมี**")
    st.caption(", ".join(FEATURES + [TARGET]))

if uploaded is None:
    st.info("👋 เริ่มต้นโดยอัปโหลดไฟล์ CSV ของ Heart Dataset ที่แถบด้านซ้าย")
    st.markdown("ไฟล์ควรมีข้อมูลตัวแปร 11 ตัว และคอลัมน์เป้าหมาย `HeartDisease` (0/1)")
    st.stop()

try:
    raw = load_csv(uploaded)
except Exception as e:
    st.error(f"อ่าน CSV ไม่สำเร็จ: {e}")
    st.stop()

data, errors = clean_data(raw)
if errors:
    for err in errors:
        st.error(err)
    st.stop()

if not set(data[TARGET].unique()).issubset({0, 1}):
    st.warning("ค่า HeartDisease ควรเป็น 0 หรือ 1 เพื่อแสดงผลกลุ่มได้ถูกต้อง")

if len(data) < 10:
    st.warning("Dataset มีข้อมูลน้อยกว่า 10 แถว ผลการประเมินอาจไม่น่าเชื่อถือ")


# Model evaluation split
X, y = data[FEATURES], data[TARGET]
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent/100, random_state=42, stratify=y)
except ValueError as e:
    st.error(f"ไม่สามารถแบ่งข้อมูล Train/Test ได้: {e}")
    st.stop()

k_train = min(k_value, len(X_train))
eval_model = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k_train))])
eval_model.fit(X_train, y_train)
y_pred = eval_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# tabs
t_dashboard, t_eval, t_predict, t_data = st.tabs(["📊 Dashboard", "🧠 Model Evaluation", "🔎 Prediction", "📁 Dataset Explorer"])

with t_dashboard:
    st.subheader("ภาพรวม Dataset")
    total = len(data)
    n0 = int((data[TARGET] == 0).sum())
    n1 = int((data[TARGET] == 1).sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ข้อมูลทั้งหมด", f"{total:,} rows")
    m2.metric("จำนวน Features", f"{len(FEATURES)}")
    m3.metric("HeartDisease = 0", f"{n0:,}", f"{n0/total*100:.1f}%")
    m4.metric("HeartDisease = 1", f"{n1:,}", f"{n1/total*100:.1f}%")
    left, right = st.columns(2)
    with left:
        st.markdown("#### สัดส่วนกลุ่มเป้าหมาย")
        fig, ax = plt.subplots(figsize=(6,3.5))
        ax.bar(["Group 0", "Group 1"], [n0, n1])
        ax.set_ylabel("Count"); ax.grid(axis="y", alpha=.2)
        fig.patch.set_facecolor("#f4f8fb"); ax.set_facecolor("#f4f8fb")
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    with right:
        st.markdown("#### การกระจายอายุ")
        fig, ax = plt.subplots(figsize=(6,3.5))
        ax.hist(data["Age"], bins=18, edgecolor="white")
        ax.set_xlabel("Age"); ax.set_ylabel("Frequency"); ax.grid(axis="y", alpha=.2)
        fig.patch.set_facecolor("#f4f8fb"); ax.set_facecolor("#f4f8fb")
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.markdown("#### สถิติเบื้องต้น")
    st.dataframe(data[FEATURES].describe().T.round(2), use_container_width=True)

with t_eval:
    st.subheader("ผลการประเมินโมเดล KNN")
    a,b,c = st.columns(3)
    a.metric("Accuracy", f"{accuracy*100:.2f}%")
    b.metric("Training rows", f"{len(X_train):,}")
    c.metric("Testing rows", f"{len(X_test):,}")
    st.markdown("#### Confusion Matrix")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5,4)); ax.imshow(cm)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    fig.patch.set_facecolor("#f4f8fb"); ax.set_facecolor("#f4f8fb")
    st.pyplot(fig); plt.close(fig)
    st.markdown("#### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
    st.markdown("#### ทดลองเปรียบเทียบค่า K")
    max_k = min(25, len(X_train))
    ks, scores = [], []
    for k in range(1, max_k+1):
        p = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k))])
        p.fit(X_train, y_train); ks.append(k); scores.append(accuracy_score(y_test, p.predict(X_test)))
    fig, ax = plt.subplots(figsize=(7,3.5)); ax.plot(ks, scores, marker="o"); ax.set_xlabel("K"); ax.set_ylabel("Accuracy"); ax.grid(alpha=.25)
    fig.patch.set_facecolor("#f4f8fb"); ax.set_facecolor("#f4f8fb")
    st.pyplot(fig, use_container_width=True); plt.close(fig)

with t_predict:
    st.subheader("ทำนายข้อมูลรายบุคคล")
    st.caption("กรอกข้อมูลให้ครบ แล้วกดปุ่มเพื่อให้โมเดล KNN จำแนกกลุ่มจากข้อมูลใกล้เคียง")
    defaults = {
        "Age": (40, 28, 77), "Sex": (1, 0, 1), "ChestPainType": (1, 1, 4), "RestingBP": (120, 80, 200),
        "Cholesterol": (200, 0, 603), "FastingBS": (0, 0, 1), "RestingECG": (1, 1, 3), "MaxHR": (150, 60, 202),
        "ExerciseAngina": (0, 0, 1), "Oldpeak": (0.0, -2.6, 6.2), "ST_Slope": (2, 1, 3)
    }
    input_cols = st.columns(3)
    values = {}
    for idx, feature in enumerate(FEATURES):
        default, low, high = defaults[feature]
        with input_cols[idx % 3]:
            if feature in ["Sex", "FastingBS", "ExerciseAngina", "ChestPainType", "RestingECG", "ST_Slope"]:
                values[feature] = st.selectbox(feature, list(range(int(low), int(high)+1)), index=int(default-low), key=f"input_{feature}")
            elif feature == "Oldpeak":
                values[feature] = st.slider(feature, float(low), float(high), float(default), 0.1, key=f"input_{feature}")
            else:
                values[feature] = st.number_input(feature, min_value=float(low), max_value=float(high), value=float(default), step=1.0, key=f"input_{feature}")
    if st.button("🔍 วิเคราะห์และจำแนกข้อมูล", use_container_width=True):
        final_model = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=min(k_value, len(data))))])
        final_model.fit(data[FEATURES], data[TARGET])
        inp = pd.DataFrame([values], columns=FEATURES)
        pred = int(final_model.predict(inp)[0])
        probs = final_model.predict_proba(inp)[0]
        classes = list(final_model.named_steps["knn"].classes_)
        confidence = float(probs[classes.index(pred)])
        if pred == 1:
            st.error("⚠️ ผลการจำแนก: HeartDisease = 1 (กลุ่ม 1)")
        else:
            st.success("✅ ผลการจำแนก: HeartDisease = 0 (กลุ่ม 0)")
        st.metric("Model confidence", f"{confidence*100:.1f}%")
        st.progress(confidence)
        st.warning("ผลนี้เป็นการจำแนกตาม Dataset เพื่อการศึกษาเท่านั้น ไม่ใช่การวินิจฉัยโรคหรือการประเมินความเสี่ยงทางการแพทย์")

with t_data:
    st.subheader("Dataset Explorer")
    st.caption(f"แสดงข้อมูลที่ผ่านการตรวจสอบและลบแถวที่มีค่าว่างแล้ว • {len(data):,} rows")
    st.dataframe(data, use_container_width=True, height=430)
    col = st.selectbox("เลือกตัวแปรเพื่อดูการกระจาย", data.columns)
    st.bar_chart(data[col].value_counts().sort_index())
    st.download_button("⬇️ ดาวน์โหลดข้อมูลที่ทำความสะอาดแล้ว", data=data.to_csv(index=False).encode("utf-8-sig"), file_name="heart_dataset_clean.csv", mime="text/csv")

st.markdown("---")
st.caption("HeartCare AI • KNN Classification Demo • Educational use only")