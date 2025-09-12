#importing libraries ------------
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_selection import chi2, SelectKBest
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import google.generativeai as genai

import streamlit as st

#DETEREMINING THE TYPE ------------
dataset2 = pd.read_csv("Headache Based Data.csv")
del dataset2["previous_attacks"]

for col in dataset2.columns:
    if dataset2[col].dtype in ["int64", "float64"]:  # numerical column
        dataset2[col].fillna(dataset2[col].mean(), inplace=True)
    else:  # categorical column
        dataset2[col].fillna(dataset2[col].mode()[0], inplace=True)

x = dataset2.drop(columns=["CLASS"])
y = dataset2["CLASS"]

#change to pericranial
x = dataset2[["headache_days", "characterisation", "nausea", "photophobia", "phonophobia", "severity", "pericranial", "aggravation", "location", "aura_duration", "rhinorrhoea", "hemiplegic", "vomitting", "conjunctival_injection", "miosis"]]
class_le = LabelEncoder()
y_encoded = class_le.fit_transform(y)
le = LabelEncoder()

le_encoders = {}
x_encoded = x.copy()

for column in x.columns:
  if x[column].dtype == "object":
    le = LabelEncoder()
    x_encoded[column] = le.fit_transform(x[column])
    le_encoders[column] = le

joblib.dump(le_encoders, 'le_encoders.joblib')

#encoding_columns ------------
numerical_column = x.select_dtypes(include=["int64", "float64"]).columns
categorical_column = x.select_dtypes(include=["object"]).columns

#training the model ------------
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

smote = SMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(x_train, y_train)

rf = RandomForestClassifier(n_estimators=250, random_state=42)
rf.fit(x_resampled, y_resampled)

#predicting with the model ------------
predicted_values = rf.predict(x_test)

#encoding/decoding the result ------------
le_encoders = joblib.load('le_encoders.joblib')

#designing the page ------------
st.set_page_config(page_title="NeuroPredict",  page_icon=None, initial_sidebar_state="collapsed")

st.markdown("""
    <h1 style='text-align: center; font-family: "DM Sans", sans-serif; color: #415457; font-size:6vw;'>
        NeuroPredict
    </h1>
""", unsafe_allow_html=True)
st.markdown("""
    <h2 style='text-align: center; font-family: "DM Sans", sans-serif; color: #768d91; font-size: 2vw;'>
        ₊˚ ✧ a headache type predictor and management advisor ✧ ₊˚
    </h2>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=Darker+Grotesque:wght@300..900&family=Manrope:wght@200..800&family=Nixie+One&family=Sora:wght@100..800&family=Stint+Ultra+Expanded&display=swap');

    .stApp {
        background-color: #F4F7F8; /*#aebbdc*/
        color: #2C3E50; /*#0d132b*/
    }
    
    h1, h2, h3, h4, h5, h6, stText {
        color: #3E8E7E;
        font-family: 'DM Sans', sans-serif;
    }
            
    p, label, .stText {
        color:#3c4e52;
        font-family: 'DM Sans', sans-serif;
    }

    .stSelectbox > div[data-baseweb="select"] > div, input[type="number"] {
        background-color: #9bb8bd !important;  /* Light mint background */
        color: #2C3E50 !important;             /* Dark text */
        border-radius: 0px;
        font-family: 'DM Sans', sans-serif;
        margin-bottom:1%;
        border: 2px dotted #2C3E50 !important; /* Dark border */
    }
    div[data-baseweb="input"] {
        border-radius: 0px !important;
        background-color: #9bb8bd !important;
        border: none !important;
        box-shadow: none !important;
    }

    div[role="combobox"] svg {
        fill: #3E8E7E !important;
    }
    
    input, select, textarea {
        border-radius: 0px !important;
    }

    /* Button Styling */
    .stButton button {
        background-color: yellow !important; /* default #6c7f82*/
        color: white !important;
        border-radius: 0px !important;
        padding: 0.6em 1.2em;
        font-size: 25px !important;
        font-family: 'DM Sans', sans-serif;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: pink; /* hover */
        color: #ffffff;
    }
    
    .stTitle, .stSubheader {
        font-family: "Stint Ultra Expanded", serif;
        text-align: center;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #d5e0e3; /* Light mint background for sidebar */
        color: #2C3E50; /* Dark text for sidebar */
        font-family: 'DM Sans', sans-serif;
        border-radius: 0px;
    }
    
    [data-testid="stInfo"] {
        background-color:transparent !important;
    }
    
    div[data-testid="stNotificationContentSuccess"] {
        background-color: #8abfa4 !important;
        border-left: 5px solid #3E8E7E !important;
        color: #2C3E50 !important;
        font-family: 'DM Sans', sans-serif;
    }
   div[data-baseweb="select"] > div {
        background-color: #9bb8bd !important;
        color: #2C3E50 !important;
        font-family: 'DM Sans', sans-serif;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.header("About")
st.sidebar.info("""
This application predicts the type of headache you may be experiencing based on your symptoms and provides tailored management advice. Please note that this tool is for informational purposes only and does not replace professional medical advice. Always consult a healthcare provider for accurate diagnosis and treatment""")

#taking user input ------------
st.markdown("""
    <style>
        .intro-box {
            background-color: #E0F2F1;  /* Light mint */
            border-left: 6px solid #3E8E7E;  /* Teal accent bar */
            padding: 16px;
            border-radius: 0px;
            color: #3c4e52;
            font-family: 'DM Sans', sans-serif;
            font-size: 16px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
            margin-bottom:2vh;
        }
    </style>

    <div class="intro-box">
        Please provide the following details about your headache.
    </div>
""", unsafe_allow_html=True)

headache_days = st.number_input("How many days in the past month have you experienced headaches?", min_value=0, max_value=30)

characterisation = st.selectbox("Could you describe your headache sensation?", ["Pressing -- like a tight band squeezing around my head", "Pulsating -- a rhythmic beating that comes and goes, sometimes with my heartbeat", "Throbbing -- a heavy, pounding sensation that feels like my head is pulsing with pressure", "Stabbing - sharp, sudden jabs of pain that feel like being poked with a needle or knife"])[0]
if characterisation == "Pressing -- like a tight band squeezing around my head":
    characterisation = "pressing"
elif characterisation == "Pulsating -- a rhythmic beating that comes and goes, sometimes with my heartbeat":
    characterisation = "pulsating"
elif characterisation == "Throbbing -- a heavy, pounding sensation that feels like my head is pulsing with pressure":
    characterisation = "throbbing"
else:
    characterisation = "stabbing"

nausea = st.selectbox("Are you experiencing any nausea?", ["yes", "no"])
photophobia = st.selectbox("Are you experiencing any sensitivity to light?", ["yes", "no"])
phonophobia = st.selectbox("Are your ears sensitive to sound during the headache?", ["yes", "no"])

severity = st.selectbox("How severe is your pain?", ["Mild -- The pain is noticeable but doesn't interfere with daily activities. You can functionally normally without needing rest or medication.", "Moderate -- The pain is uncomfortable and distracting. It may interfere with your concentration or daily tasks, and you may feel the need to lie down or take medication.", "Severe -- The pain is intense and debilitating. It significantly limits your ability to perform daily activities, and you likely need to rest in a dark, quiet room and take strong medication to manage the pain."])[0]
if severity == "Mild -- The pain is noticeable but doesn't interfere with daily activities. You can functionally normally without needing rest or medication.":
    severity = "mild"
elif severity == "Moderate -- The pain is uncomfortable and distracting. It may interfere with your concentration or daily tasks, and you may feel the need to lie down or take medication.":
    severity = "moderate"
else:
    severity = "severe"

pericranial = st.selectbox("Do the muscles around your temples, neck, or scalp feel tender or sore?", ["yes", "no"])
aggravation = st.selectbox("Does anything like movement, noise, or light make your headache worse?", ["yes", "no"])

location= st.selectbox("Where is your headache pain mainly located?", ["Both sides (bilateral)", "Forehead", "Around or behind the eye (orbital)", "One side only (unilateral)"])[0]
if location == "Both sides (bilateral)":
    location = "bilateral"
elif location == "Forehead":
    location = "forehead"
elif location == "Around or behind the eye (orbital)":
    location = "orbital"
else:
    location = "unilateral"

aura_duration= st.selectbox("If you experience visual or sensory disturbances (aura) before your headache, how long do they usually last?", ["No aura", "Around 5-30 minutes", "Around 1-4 hours", "4+ hours, a day, or longer"])[0]
if aura_duration == "No aura":
    aura_duration = "none"
elif aura_duration == "Around 5-30 minutes": 
    aura_duration = "minutes"
elif aura_duration == "Around 1-4 hours":
    aura_duration = "hour"
else:
    aura_duration = "day"

rhinorrhoea = st.selectbox("Do you have a runny nose with your headache?", ["yes", "no"])
hemiplegic = st.selectbox("Have you experienced temporary weakness or paralysis on one side of your body during the headache?", ["yes", "no"])
vomitting = st.selectbox("Have you vomitted during this headache episode?", ["yes", "no"])
conjunctival_injection = st.selectbox("Have the whites of your eyes appeared red or bloodshot during the headache?", ["yes", "no"])
miosis = st.selectbox("Have you noticed one of your pupils becoming unusually small?", ["yes", "no"])

user_inputs = {
    "headache_days": headache_days,
    "characterisation": characterisation,
    "nausea": nausea,
    "photophobia": photophobia,
    "phonophobia": phonophobia,
    "severity": severity,
    "pericranial": pericranial,
    "aggravation": aggravation,
    "location": location,
    "aura_duration": aura_duration,
    "rhinorrhoea": rhinorrhoea,
    "hemiplegic": hemiplegic,
    "vomitting": vomitting,
    "conjunctival_injection": conjunctival_injection,
    "miosis": miosis
}

st.markdown("""
    <style>
        .thank-you-box {
            background-color: #E0F2F1;  /* Light mint */
            border-left: 6px solid #3E8E7E;  /* Teal accent bar */
            padding: 16px;
            margin-top: 20px;
            border-radius: 0px;
            color: #3c4e52;
            font-family: 'DM Sans', sans-serif;
            font-size: 16px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
            margin-bottom:3vh;
        }
    </style>

    <div class="thank-you-box">
        Thank you for providing the details. Click the button below to get your headache type prediction and management advice.
    </div>
""", unsafe_allow_html=True)


if st.button("Predict Headache Type and Get Advice"):
    user_df = pd.DataFrame([user_inputs])
    user_encoded = user_df.copy()

    for column, le in le_encoders.items():
        if column in user_encoded.columns:
            user_encoded[column] = le.transform(user_encoded[column])

    #revealing the prediction ------------
    prediction = rf.predict(user_encoded)
    decoded_prediction = class_le.inverse_transform([prediction[0]])[0]

    #migraine --> 1, sinus --> 2, tension --> 3, cluster --> 0
    st.subheader("Predicted Headache Type:")
    st.success(f"{decoded_prediction.capitalize()}")


    #LLM PART OF THE PROJECT -----------------------------
    genai.configure(api_key="AIzaSyAQkVDq7TLT1D24klQGHW0oaPZMxsfTxT8")
    model=genai.GenerativeModel("gemini-2.5-flash")

    prompt = f'''
    Explain {decoded_prediction} Generate a clear, structured, and medically relevant response that does not stray from the topic of headache management. The output must include: (1) a list of home remedies tailored to the identified headache type, starting with simple and commonly manageable options, then progressing to more intensive measures only if appropriate; (2) a set of official or medically recognized remedies and management strategies, organized from standard/common recommendations to those used for severe cases; and (3) a list of recommended medications suitable for {decoded_prediction}, with a clear distinction between over-the-counter medicines and prescription-only options available at a pharmacy. All information should be concise, practical, and directly related to the headache type, avoiding unrelated advice or general health information. Remedies and medications must be presented with brief explanations or usage notes where relevant to ensure clarity, while avoiding speculation or unverified treatments.
    '''

    with st.spinner('Generating management advice...'):
        response=model.generate_content(prompt)

    st.subheader(f"Management Advice for {decoded_prediction.capitalize()}:")
    st.write(response.text)

#streamlit run MigraineProject.py
