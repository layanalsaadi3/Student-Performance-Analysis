"""
Student Performance Predictor - Streamlit Application
واجهة مستخدم مبسطة ومباشرة للطلاب لتوقع درجة الاختبار
باللغة العربية بالكامل مع محاذاة من اليمين إلى اليسار (RTL)
"""

import os
import joblib
import pandas as pd
import streamlit as st

# ==============================================================================
# إعدادات الصفحة والتصميم العصري (Page Configuration & Styling)
# ==============================================================================
st.set_page_config(
    page_title="حاسبة توقع درجة الاختبار للطلاب",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# تخصيص المظهر بالكامل لدعم اللغة العربية من اليمين لليسار (RTL)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800;900&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Tajawal', sans-serif !important;
        direction: rtl !important;
        text-align: right !important;
    }

    /* رأس الصفحة */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        direction: rtl;
    }
    
    .app-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1D4ED8 0%, #7C3AED 50%, #059669 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .app-subtitle {
        font-size: 1.1rem;
        color: #475569;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }
    
    /* بطاقة النتيجة */
    .score-card {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.08) 0%, rgba(124, 58, 237, 0.08) 100%);
        border: 2px solid rgba(37, 99, 235, 0.25);
        border-radius: 20px;
        padding: 26px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.12);
        direction: rtl;
    }
    
    .score-number {
        font-size: 4rem;
        font-weight: 900;
        color: #1E40AF;
        line-height: 1.1;
        margin: 10px 0;
    }
    
    .score-label {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1E293B;
    }
    
    .badge {
        display: inline-block;
        padding: 6px 20px;
        border-radius: 9999px;
        font-size: 1rem;
        font-weight: 700;
        margin-top: 10px;
    }
    
    .badge-a { background: #DCFCE7; color: #166534; border: 1px solid #86EFAC; }
    .badge-b { background: #DBEAFE; color: #1E40AF; border: 1px solid #93C5FD; }
    .badge-c { background: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; }
    .badge-d { background: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }
    
    /* حاويات الأقسام */
    .input-section {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(226, 232, 240, 0.35);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 22px;
        direction: rtl;
        text-align: right;
    }
    
    .section-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
        direction: rtl;
        text-align: right;
    }

    /* ضبط أشرطة التمرير والمدخلات بالكامل لليمين (RTL) */
    div[data-testid="stSlider"],
    div[data-testid="stSelectbox"],
    div[data-testid="stWidgetLabel"] {
        direction: rtl !important;
        text-align: right !important;
    }

    div[data-testid="stSlider"] label,
    div[data-testid="stSelectbox"] label {
        direction: rtl !important;
        text-align: right !important;
        display: block !important;
        width: 100% !important;
        margin-bottom: 6px !important;
    }

    div[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p,
    div[data-testid="stSelectbox"] [data-testid="stMarkdownContainer"] p {
        text-align: right !important;
        direction: rtl !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #1E293B !important;
    }

    /* محاذاة كتل الأعمدة لليمين */
    div[data-testid="stHorizontalBlock"] {
        direction: rtl !important;
    }

    div[data-testid="column"] {
        direction: rtl !important;
        text-align: right !important;
    }

    /* القوائم المنسدلة */
    div[data-baseweb="select"] {
        direction: rtl !important;
        text-align: right !important;
    }
    
    div[data-baseweb="popover"] {
        direction: rtl !important;
        text-align: right !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# تحميل النموذج المدرب (Load Pretrained ML Model)
# ==============================================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_student_predictor.joblib")

@st.cache_resource
def load_predictor():
    """يحمل نموذج التنبؤ المدرب، أو يقوم بتدريبه وحفظه إذا لم يكن موجوداً."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        from train import run_full_pipeline
        results = run_full_pipeline()
        return results["best_pipeline"]

model = load_predictor()

# ==============================================================================
# رأس الصفحة (Header)
# ==============================================================================
st.markdown("""
<div class="main-header">
    <div class="app-title">🎓 حاسبة توقع درجة الاختبار</div>
    <div class="app-subtitle">اسحب الشريط وحدد بياناتك الدراسية لمعرفة درجتك المتوقعة في الاختبار من 0 إلى 100</div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# مدخلات الطالب عبر أشرطة السحب (Student Sliders - All in Arabic & RTL)
# ==============================================================================
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📚 عادات المذاكرة والتحصيل الأكاديمي</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    hours_studied = st.slider(
        "ساعات المذاكرة أسبوعياً",
        min_value=1,
        max_value=44,
        value=20,
        help="اسحب الشريط لاختيار عدد ساعات المذاكرة والمراجعة الأسبوعية"
    )
    
    attendance = st.slider(
        "نسبة الحضور المدرسي (%)",
        min_value=60,
        max_value=100,
        value=85,
        help="اسحب الشريط لاختيار نسبة حضورك للحصص الدراسية"
    )

with col2:
    previous_scores = st.slider(
        "معدل درجاتك السابقة (%)",
        min_value=50,
        max_value=100,
        value=75,
        help="اسحب الشريط لتحديد متوسط درجاتك في الاختبارات السابقة"
    )

    tutoring_sessions = st.slider(
        "عدد جلسات التقوية شهرياً",
        min_value=0,
        max_value=8,
        value=2,
        help="اسحب الشريط لاختيار عدد حصص التقوية أو الدروس الإضافية في الشهر"
    )

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🌱 نمط الحياة ومستوى الحماس</div>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)

with col3:
    sleep_hours = st.slider(
        "ساعات النوم يومياً",
        min_value=4,
        max_value=10,
        value=7,
        help="اسحب الشريط لاختيار متوسط ساعات نومك كل ليلة"
    )

with col4:
    motivation_options = {
        "عالي": "High",
        "متوسط": "Medium",
        "منخفض": "Low"
    }
    motivation_display = st.selectbox(
        "مستوى التحفيز الدراسي",
        options=list(motivation_options.keys()),
        index=1,
        help="اختر مدى حماسك ورغبتك في التفوق الدراسي"
    )
    motivation_level = motivation_options[motivation_display]

with col5:
    activity_options = {
        "نعم": "Yes",
        "لا": "No"
    }
    activity_display = st.selectbox(
        "المشاركة في الأنشطة اللاصفية",
        options=list(activity_options.keys()),
        index=0,
        help="هل تشارك في الأنشطة الرياضية أو المدرسية اللاصفية؟"
    )
    extracurricular = activity_options[activity_display]

st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# حساب وتوقع النتيجة (Prediction Execution)
# ==============================================================================
input_df = pd.DataFrame([{
    "Hours_Studied": hours_studied,
    "Attendance": attendance,
    "Previous_Scores": previous_scores,
    "Sleep_Hours": sleep_hours,
    "Tutoring_Sessions": tutoring_sessions,
    "Motivation_Level": motivation_level,
    "Extracurricular_Activities": extracurricular
}])

predicted_raw = float(model.predict(input_df)[0])
# حصر النتيجة بدقة بين 0 و 100
predicted_score = min(100.0, max(0.0, predicted_raw))

# تحديد التقدير واللون بالعربي
if predicted_score >= 80:
    grade_label = "ممتاز (أ) — أداء دراسي متفوق جداً"
    badge_class = "badge-a"
elif predicted_score >= 70:
    grade_label = "جيد جداً (ب) — مستوى قوي"
    badge_class = "badge-b"
elif predicted_score >= 60:
    grade_label = "جيد (ج) — مستوى مستقر"
    badge_class = "badge-c"
else:
    grade_label = "يحتاج إلى دعم (د) — ركّز على زيادة الحضور والمذاكرة"
    badge_class = "badge-d"

# ==============================================================================
# بطاقة النتيجة للمستخدم (Display Result)
# ==============================================================================
st.markdown(f"""
<div class="score-card">
    <div class="score-label">درجتك المتوقعة في الاختبار</div>
    <div class="score-number">{predicted_score:.1f} <span style="font-size: 1.8rem; color: #64748B;">من 100</span></div>
    <div><span class="badge {badge_class}">{grade_label}</span></div>
</div>
""", unsafe_allow_html=True)

# نصيحة توجيهية ذكية مبسطة
if attendance < 80:
    st.info("💡 **نصيحة مهمة:** زيادة نسبة حضورك إلى 90% فأعلى هي أسرع وسيلة لرفع درجتك المتوقعة بمقدار 2 إلى 4 درجات!")
elif hours_studied < 18:
    st.info("💡 **نصيحة للمذاكرة:** زيادة ساعتين أو ثلاث ساعات إضافية أسبوعياً في المذاكرة المركزة سترفع من نتيجتك بشكل ملحوظ.")
else:
    st.success("🌟 **أحسنت!** نمطك الدراسي متوازن وممتاز، والاستمرار على هذا الالتزام يضمن لك التفوق بإذن الله.")

# تذييل الصفحة
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.85rem; margin-top: 30px; direction: rtl;">
    حاسبة توقع درجات الطلاب • مدعومة بنموذج تعلم الآلة (Machine Learning)
</div>
""", unsafe_allow_html=True)
