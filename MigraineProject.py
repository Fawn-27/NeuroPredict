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

genai.configure(api_key="AIzaSyAQkVDq7TLT1D24klQGHW0oaPZMxsfTxT8")
model=genai.GenerativeModel("gemini-2.5-flash")

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
st.set_page_config(page_title="NeuroPredict",  page_icon=None, initial_sidebar_state="collapsed", layout="centered")

st.markdown("""
    <h1 style='text-align: center; margin-bottom:0px; font-family: "Cormorant", serif; font-style: italic; color: #0f4662; font-size:7vw;'>✧NeuroPredict✧</h1>
    <h2 style='text-align: center; margin-top:0px;font-family: "Cormorant", serif; color: #637f8b; font-style:italic;font-size: 3vw;'>your headache type predictor and management advisor</h2>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant:ital,wght@0,300..700;1,300..700&family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=Darker+Grotesque:wght@300..900&family=Manrope:wght@200..800&family=Nixie+One&family=Sora:wght@100..800&family=Stint+Ultra+Expanded&display=swap');

    .stApp {
        background-color: #f8f8f8; /*#aebbdc*/
        color: #0f4662; /*#0d132b*/
    }
    
    h1, h2, h3, h4, h5, h6, stText {
        color: #0f4662;
        font-family: 'Manrope', sans-serif;
    }
            
    p, label, stText {
        color:#3c4e52;
        font-family: 'Manrope', sans-serif;
    }

    .stSelectbox > div[data-baseweb="select"] > div, input[type="number"] {
        background-color: #dbe5ea !important;  
        color: #0f4662;         
        /*border-radius: 0px;*/
        font-family: 'Manrope', sans-serif;
        margin-bottom:1%;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #dbe5ea; /* Light mint background for sidebar */
        color: #0f4662; /* Dark text for sidebar */
        font-family: 'Manrope', sans-serif;
    }
            
    section[data-testid="stSidebar"] [data-widget-key="title_sidebar"] {
        font-family: 'Manrope', sans-serif !important;
    }
            
    [data-testid="stInfo"] {
        background-color:transparent !important;
    }
    
    div[data-testid="stNotificationContentSuccess"] {
        background-color: #a9becb !important;
        border-left: 10px dotted #465e56 !important;
        color: #0f4662 !important;
        font-family: 'Manrope', sans-serif;
    }
            
   div[data-baseweb="select"] > div {
        background-color: #7994a0 !important;
        color: #0f4662 !important;
        font-family: 'Manrope', sans-serif;
        font-size: 16px;
        border-bottom: 4px solid #0f4662;
    }

    .stButton > button:hover {
        background-color: #7994a0 !important; /* Darker on hover */
        font_size:26px;
        color:white !important;
    }
            
    </style>
    """, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* Target sidebar header */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 , section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6, section[data-testid="stSidebar"] p {
        color: #0f4662 !important;  
        font-family: 'Manrope', sans-serif !important;
        text-align:left !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#taking user input ------------
st.markdown("""
    <style>
        .intro-box {
            background-color: #a9becb;  /* Light mint */
            border-left: 6px solid #0f4662;  /* Teal accent bar */
            padding: 16px;
            border-radius: 0px;
            color: #0f4662;
            font-family: 'Manrope', sans-serif;
            font-size: 16px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
            margin-bottom:2vh;
            text-align:center;
        }

    </style>

    <div class="intro-box">
        Please provide the following details about your headache. If you don't select an option, it'll be considered as the first option.
    </div>
""", unsafe_allow_html=True)



headache_days = st.number_input("How many days in the past month have you experienced headaches?", min_value=0, max_value=30, key="headache_days")


characterisation_options = [
    "-- Select an option --",
    "Pressing -- like a tight band squeezing around my head",
    "Pulsating -- a rhythmic beating that comes and goes, sometimes with my heartbeat",
    "Throbbing -- a heavy, pounding sensation that feels like my head is pulsing with pressure",
    "Stabbing - sharp, sudden jabs of pain that feel like being poked with a needle or knife"
]
characterisation = st.selectbox("What best describes how your headache feels", characterisation_options, index=0)
if characterisation == characterisation_options[0]:
    characterisation="pressing"
elif "Pressing" in characterisation:
    characterisation = "pressing"
elif "Pulsating" in characterisation:
    characterisation = "pulsating"
elif "Throbbing" in characterisation:
    characterisation = "throbbing"
elif "Stabbing" in characterisation:
    characterisation = "stabbing"

nausea_options = ["-- Select an option --", "Yes", "No"]
nausea = st.selectbox("Are you experiencing any nausea?", nausea_options, index=0)
if nausea == nausea_options[0]:
    nausea="yes"
elif "Yes" in nausea:
    nausea = "yes"
elif "No" in nausea:
    nausea = "no"

photophobia_options = ["-- Select an option --", "Yes", "No"]
photophobia = st.selectbox("Are you experiencing any sensitivity to light?", photophobia_options, index=0)
if photophobia == photophobia_options[0]:
    photophobia="yes"
elif "Yes" in photophobia:
    photophobia = "yes"
elif "No" in photophobia:
    photophobia = "no"

phonophobia_options = ["-- Select an option --", "Yes", "No"]
phonophobia = st.selectbox("Are your ears sensitive to sound during the headache?", phonophobia_options, index=0)
if phonophobia == phonophobia_options[0]:
    phonophobia="yes"
elif "Yes" in phonophobia:
    phonophobia = "yes"
elif "No" in phonophobia:
    phonophobia = "no"

severity_options = ["-- Select an option --", "Mild -- The pain is noticeable but doesn't interfere with daily activities. You can functionally normally without needing rest or medication.", "Moderate -- The pain is uncomfortable and distracting. It may interfere with your concentration or daily tasks, and you may feel the need to lie down or take medication.", "Severe -- The pain is intense and debilitating. It significantly limits your ability to perform daily activities, and you likely need to rest in a dark, quiet room and take strong medication to manage the pain."]
severity = st.selectbox("How severe is your pain?", severity_options, index=0)[0]
if severity == severity_options[0]:
    severity="mild"
elif severity == "Mild -- The pain is noticeable but doesn't interfere with daily activities. You can functionally normally without needing rest or medication.":
    severity = "mild"
elif severity == "Moderate -- The pain is uncomfortable and distracting. It may interfere with your concentration or daily tasks, and you may feel the need to lie down or take medication.":
    severity = "moderate"
else:
    severity = "severe"

pericranial_options = ["-- Select an option --", "Yes", "No"]
pericranial = st.selectbox("Do the muscles around your temples, neck, or scalp feel tender or sore?", pericranial_options, index=0)
if pericranial == pericranial_options[0]:
    pericranial="yes"
elif "Yes" in pericranial:
    pericranial = "yes"
elif "No" in pericranial:
    pericranial = "no"

aggravation_options = ["-- Select an option --", "Yes", "No"]   
aggravation = st.selectbox("Does anything like movement, noise, or light make your headache worse?", aggravation_options, index=0)
if aggravation == aggravation_options[0]:
    aggravation="yes"
elif "Yes" in aggravation:
    aggravation = "yes"
elif "No" in aggravation:
    aggravation = "no"

location_options = ["-- Select an option --", "Both sides (bilateral)", "Forehead", "Around or behind the eye (orbital)", "One side only (unilateral)"]
location= st.selectbox("Where is your headache pain mainly located?", location_options, index=0)[0]
if location == location_options[0]:
    location="bilateral"
elif location == "Both sides (bilateral)":
    location = "bilateral"
elif location == "Forehead":
    location = "forehead"
elif location == "Around or behind the eye (orbital)":
    location = "orbital"
else:
    location = "unilateral"

aura_duration_options = ["-- Select an option --", "No aura", "Around 5-30 minutes", "Around 1-4 hours", "4+ hours, a day, or longer"]
aura_duration= st.selectbox("If you experience visual or sensory disturbances (aura) before your headache, how long do they usually last?", aura_duration_options, index=0)[0]
if aura_duration == aura_duration_options[0]:
    aura_duration="none"
elif aura_duration == "No aura":
    aura_duration = "none"
elif aura_duration == "Around 5-30 minutes": 
    aura_duration = "minutes"
elif aura_duration == "Around 1-4 hours":
    aura_duration = "hour"
else:
    aura_duration = "day"

rhinorrhoea_options = ["-- Select an option --","Yes", "No"]
rhinorrhoea = st.selectbox("Do you have a runny nose with your headache?", rhinorrhoea_options, index=0)
if rhinorrhoea == rhinorrhoea_options[0]: 
    rhinorrhoea="yes"
elif "Yes" in rhinorrhoea:
    rhinorrhoea = "yes"
elif "No" in rhinorrhoea:
    rhinorrhoea = "no"

hemiplegic_options = ["-- Select an option --", "Yes", "No"]
hemiplegic = st.selectbox("Have you experienced temporary weakness or paralysis on one side of your body during the headache?", hemiplegic_options, index=0)
if hemiplegic == hemiplegic_options[0]:
    hemiplegic="yes"
elif "Yes" in hemiplegic:
    hemiplegic = "yes"
elif "No" in hemiplegic:
    hemiplegic = "no"

vomitting_options = ["-- Select an option --", "Yes", "No"]
vomitting = st.selectbox("Have you vomitted during your headache?", vomitting_options, index=0)
if vomitting == vomitting_options[0]:
    vomitting="yes"
elif "Yes" in vomitting:
    vomitting = "yes"
elif "No" in vomitting:
    vomitting = "no"

conjunctival_injection_options = ["-- Select an option --", "Yes", "No"]
conjunctival_injection = st.selectbox("Have the whites of your eyes appeared red or bloodshot during the headache?", conjunctival_injection_options, index=0)
if conjunctival_injection == conjunctival_injection_options[0]:
    conjunctival_injection="yes"
elif "Yes" in conjunctival_injection:
    conjunctival_injection = "yes"
elif "No" in conjunctival_injection:
    conjunctival_injection = "no"

miosis_options = ["-- Select an option --", "Yes", "No"]
miosis = st.selectbox("Have you noticed one of your pupils becoming unusually small?", miosis_options, index=0)
if miosis == miosis_options[0]:
    miosis="yes"
elif "Yes" in miosis:
    miosis = "yes"
elif "No" in miosis:
    miosis = "no"

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
            background-color: #a9becb;  /* Light mint */
            border-left: 6px solid #0f4662;  /* Teal accent bar */
            padding: 16px;
            margin-top: 20px;
            border-radius: 0px;
            color: #0f4662;
            font-family: 'Manrope', sans-serif;
            font-size: 16px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
            margin-bottom:3vh;
            text-align:center;
        }
    </style>

    <div class="thank-you-box">
        Thank you! Click the button below to get your headache type prediction and management advice.
    </div>
""", unsafe_allow_html=True)

st.markdown("""<style>
            .stButton button {
        background-color: #a9becb !important; 
        color: white !important;
        padding: 0.6em 1.2em;
        font-size: 25px !important;
        font-family: 'Manrope', sans-serif;
        transition: background-color 0.3s ease;
            display:block;
            margin:auto;
    }

            </style>
""", unsafe_allow_html=True)

if st.button("Predict Headache Type and Get Advice", key="predict_button"):
    user_df = pd.DataFrame([user_inputs])
    user_encoded = user_df.copy()

    for column, le in le_encoders.items():
        if column in user_encoded.columns:
            user_encoded[column] = le.transform(user_encoded[column])

    #revealing the prediction ------------
    prediction = rf.predict(user_encoded)
    decoded_prediction = class_le.inverse_transform([prediction[0]])[0]

    #migraine --> 1, sinus --> 2, tension --> 3, cluster --> 0
    st.markdown("""<h2 style='text-align: center; margin-bottom:0px;font-family: "Cormorant", serif; color: #637f8b; font-style:italic;font-size: 1.5vw;'>━━━━⊱⋆⊰━━━━</h2>""", unsafe_allow_html=True)
    st.markdown("""
    <h2 style='text-align: center; margin-bottom:0px;font-family: "Cormorant", serif; color: #637f8b; font-style:italic;font-size: 3vh;'>Predicted Headache Type:</h2>""", unsafe_allow_html=True)
    st.success(f"{decoded_prediction.capitalize()}")

    #LLM PART OF THE PROJECT -----------------------------
    prompt = f'''
    Explain {decoded_prediction} Generate a clear, structured, and medically relevant response that does not stray from the topic of headache management. The output must include: (1) a list of home remedies tailored to the identified headache type, starting with simple and commonly manageable options, then progressing to more intensive measures only if appropriate; (2) a set of official or medically recognized remedies and management strategies, organized from standard/common recommendations to those used for severe cases; and (3) a list of recommended medications suitable for {decoded_prediction}, with a clear distinction between over-the-counter medicines and prescription-only options available at a pharmacy. All information should be concise, practical, and directly related to the headache type, avoiding unrelated advice or general health information. Remedies and medications must be presented with brief explanations or usage notes where relevant to ensure clarity, while avoiding speculation or unverified treatments.
    '''

    with st.spinner('Generating management advice...'):
        response=model.generate_content(prompt)

    st.subheader(f"Management Advice for {decoded_prediction.capitalize()}:")
    
    st.write(response.text)

placeholder = st.sidebar.empty()

st.sidebar.header("About")
st.sidebar.write("""
This application predicts the type of headache you may be experiencing based on your symptoms and provides tailored management advice. Please note that this tool is for informational purposes only and does not replace professional medical advice. Always consult a healthcare provider for accurate diagnosis and treatment""")

st.sidebar.header("Developer")
st.sidebar.write("""Hi, I'm Mansi, a high school student passionate about the intersection of cogntiive science, healthcare, and AI. This project combines machine learning and large language models to help people understand and manage headaches better. Feel free to reach out if you have any questions or feedback!""")

st.sidebar.header("Lifehacks!")
st.sidebar.markdown("""
Want to avoid headaches? Here are 3 top lifehacks for a healthy life:
""", unsafe_allow_html=True)

life_hacks = f'''
    List 3 science-backed, practical lifehack to prevent headaches and support a healthy lifestyle—clear, engaging/#entertaining, and easy to apply daily.
'''
with st.sidebar:
    with st.spinner("Generating life hacks.."):
        hacks=model.generate_content(life_hacks)

st.sidebar.write(hacks.text)
