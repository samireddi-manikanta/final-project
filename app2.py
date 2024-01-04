import streamlit as st
import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the pest identification model
loadedmodel = tf.keras.models.load_model('mbv2model1_split')
p = ["Bollworms", "Fallarmyworms", "Thrips", "aphids", "black cutworms"]
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
font_color = (255, 0, 0)
thickness = 2
threshold = 0.8

# Load the fertilizer recommendation model
loaded_fertilizer_model = joblib.load('fertilizer_recommender.pkl')
#fertilizer_dataset = pd.read_csv('fertilizerdataset.csv')
#le = LabelEncoder()
#y = fertilizer_dataset['Fertilizer']
#y_encoded = le.fit_transform(y)

# Define the recommend_insecticide function
def recommend_insecticide(pest):
    if pest == "No pests":
        return "No pest detected, Happy Farming..."
    insecticides_data = pd.read_excel("Pesticides.xlsx")
    recommended_insecticides = insecticides_data[insecticides_data["Pest"] == pest]
    recommendation_text = f"Recommended insecticides for {pest}:\n"
    biological_options = recommended_insecticides["Biological Management"].tolist()
    dosage_bio = recommended_insecticides["Dosage (Biological)"].tolist()
    if biological_options:
        recommendation_text += f"\n**Biological Options:**\n"
        for option, dosage in zip(biological_options, dosage_bio):
            recommendation_text += f"- {option} ({dosage})\n"
    chemical_options = recommended_insecticides["Chemical Management"].tolist()
    dosage_chem = recommended_insecticides["Dosage (Chemical)"].tolist()
    if chemical_options:
        recommendation_text += f"\n**Chemical Options:**\n"
        for option, dosage in zip(chemical_options, dosage_chem):
            recommendation_text += f"- {option} ({dosage})\n"
    recommendation_text += "\n**Additional Recommendations:**\n"
    recommendation_text += "- Always consult with a qualified pest control professional before applying any insecticides.\n"
    recommendation_text += "- Follow safety guidelines and wear appropriate protective equipment when handling insecticides.\n"
    recommendation_text += "- Consider rotating insecticides to prevent pest resistance.\n"
    return recommendation_text

st.title("Pest Identification and Farming Recommendations")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_frame = cv2.resize(image, (224, 224))
    processed_frame = processed_frame / 255.0
    processed_frame = np.expand_dims(processed_frame, axis=0)
    prediction = loadedmodel.predict(processed_frame)
    pests_detected = []
    ipest = ""
    for i, pred in enumerate(prediction[0]):
        if pred >= threshold:
            pests_detected.append(f"Pest {p[i]}: {pred}")
            ipest = p[i]
    if pests_detected:
        st.subheader("Pests Detected:")
        for pest in pests_detected:
            st.write(pest)

        insecticide_recommendation = recommend_insecticide(ipest)
        st.subheader("Insecticide Recommendation:")
        st.write(insecticide_recommendation)

        st.subheader("Fertilizer Recommendation:")
        Nitrogen_value = st.number_input("Enter Nitrogen value:")
        Phosphorus_value = st.number_input("Enter Phosphorus value:")
        Potassium_value = st.number_input("Enter Potassium value:")
        pH_value = st.number_input("Enter pH value:")
        Rainfall_value = st.number_input("Enter Rainfall value:")
        Temperature_value = st.number_input("Enter Temperature value:")

        # Add Predict button for fertilizer recommendation
        if st.button("Predict Fertilizer"):
            new_input = [Nitrogen_value, Phosphorus_value, Potassium_value, pH_value, Rainfall_value, Temperature_value]
            recommended_fertilizer_encoded = loaded_fertilizer_model.predict([new_input])[0]
            #recommended_fertilizer = le.inverse_transform([recommended_fertilizer_encoded])[0]
            st.write("Recommended fertilizer:", recommended_fertilizer_encoded)
    else:
        st.write("No pests detected.")
