from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os

app = Flask(__name__)

# Load the data and train the model
df = pd.read_csv(os.path.abspath(os.path.dirname(__file__)) + "/Training.csv")
df.columns = df.columns.str.strip()
df['prognosis'] = df['prognosis'].str.strip()

# Correct known typos
df['prognosis'] = df['prognosis'].replace({
    'Peptic ulcer diseae': 'Peptic ulcer disease',
    'Dimorphic hemmorhoids(piles)': 'Dimorphic hemorrhoids(piles)',
    'Osteoarthristis': 'Osteoarthritis',
    'Diabetes ': 'Diabetes'
})

prognosis_dict = {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
                  'Peptic ulcer disease': 5, 'AIDS': 6, 'Diabetes': 7, 'Gastroenteritis': 8, 
                  'Bronchial Asthma': 9, 'Hypertension': 10, 'Migraine': 11, 'Cervical spondylosis': 12,
                  'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16, 
                  'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19, 'Hepatitis B': 20, 'Hepatitis C': 21, 
                  'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic hepatitis': 24, 'Tuberculosis': 25, 
                  'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemorrhoids(piles)': 28, 'Heart attack': 29, 
                  'Varicose veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32, 'Hypoglycemia': 33, 
                  'Osteoarthritis': 34, 'Arthritis': 35, '(vertigo) Paroymsal  Positional Vertigo': 36, 
                  'Acne': 37, 'Urinary tract infection': 38, 'Psoriasis': 39, 'Impetigo': 40}

# Replace prognosis names with numbers
df['prognosis'] = df['prognosis'].map(prognosis_dict)

# Check for any unmapped values
if df['prognosis'].isnull().any():
    raise ValueError("There are unmapped prognosis values in the dataset.")

l1 = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 
      'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 
      'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 
      'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 
      'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 
      'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 
      'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 
      'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 
      'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 
      'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 
      'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 
      'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 
      'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 
      'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 
      'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 
      'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 
      'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 
      'foul_smell_of_urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 
      'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 
      'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic_patches', 
      'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 
      'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 
      'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 
      'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf', 
      'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 
      'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 
      'red_sore_around_nose', 'yellow_crust_ooze']

X = df[l1]
y = df["prognosis"]

# Ensure that the target variable y is of integer type
y = y.astype(int)

if X.isnull().values.any() or y.isnull().values.any():
    raise ValueError("Dataset contains missing values. Please clean the data.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

clf3 = RandomForestClassifier(random_state=42)
clf3.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html', symptoms=l1)

@app.route('/predict', methods=['POST'])
def predict():
    sym1 = request.form.get('symptom1')
    sym2 = request.form.get('symptom2')
    sym3 = request.form.get('symptom3')
    sym4 = request.form.get('symptom4')
    sym5 = request.form.get('symptom5')

    # Debugging
    print(f"Received symptoms: {sym1}, {sym2}, {sym3}, {sym4}, {sym5}")

    symptoms = [sym1, sym2, sym3, sym4, sym5]
    l2 = [0] * len(l1)
    for symptom in symptoms:
        if symptom in l1:
            l2[l1.index(symptom)] = 1
    
    # Debugging: Print input vector
    print(f"Input feature vector: {l2}")

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted = predict[0]

    result_disease = ""
    for disease_name, index in prognosis_dict.items():
        if predicted == index:
            result_disease = disease_name
            break

    # Debugging
    print(f"Predicted disease: {result_disease}")

    return render_template('result.html', disease=result_disease)

if __name__ == '__main__':
    app.run(debug=True)
