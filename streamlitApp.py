import streamlit as st
from src.mushroom_classification.pipelines.prediction_pipeline import CustomData, PredictPipeline
import os
from src.mushroom_classification.utils.utils import load_object
import pandas as pd



# Streamlit app
st.title("Mushroom Classification App")

# Define options for each feature
feature_options = {
    "cap-shape": {"b": "Bell", "c": "Conical", "x": "Convex", "f": "Flat", "k": "Knobbed", "s": "Sunken"},
    "cap-surface": {"f": "Fibrous", "g": "Grooves", "y": "Scaly", "s": "Smooth"},
    "cap-color": {"n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "r": "Green", "p": "Pink", "u": "Purple", "e": "Red", "w": "White", "y": "Yellow"},
    "bruises": {"t": "Bruises", "f": "No"},
    "odor": {"a": "Almond", "l": "Anise", "c": "Creosote", "y": "Fishy", "f": "Foul", "m": "Musty", "p": "Pungent", "s": "Spicy", "n": "None"},
    "gill-attachment": {"a": "Attached", "d": "Descending", "f": "Free", "n": "Notched"},
    "gill-spacing": {"c": "Close", "w": "Crowded", "d": "Distant"},
    "gill-size": {"b": "Broad", "n": "Narrow"},
    "gill-color": {"k": "Black", "n": "Brown", "b": "Buff", "h": "Chocolate", "g": "Gray", "r": "Green", "o": "Orange", "p": "Pink", "u": "Purple", "e": "Red", "w": "White", "y": "Yellow"},
    "stalk-shape": {"e": "Enlarging", "t": "Tapering"},
    "stalk-root": {"b": "Bulbous", "c": "Club", "u": "Cup", "e": "Equal", "z": "Rhizomorphs", "r": "Rooted", "m": "Missing"},
    "stalk-surface-above-ring": {"f": "Fibrous", "y": "Scaly", "k": "Silky", "s": "Smooth"},
    "stalk-surface-below-ring": {"f": "Fibrous", "y": "Scaly", "k": "Silky", "s": "Smooth"},
    "stalk-color-above-ring": {"n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "o": "Orange", "p": "Pink", "e": "Red", "w": "White", "y": "Yellow"},
    "stalk-color-below-ring": {"n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "o": "Orange", "p": "Pink", "e": "Red", "w": "White", "y": "Yellow"},
    "veil-type": {"p": "Partial", "u": "Universal"},
    "veil-color": {"n": "Brown", "o": "Orange", "w": "White", "y": "Yellow"},
    "ring-number": {"o": "One", "t": "Two", "n": "None"},
    "ring-type": {"c": "Cobwebby", "e": "Evanescent", "f": "Flaring", "l": "Large", "p": "Pendant", "s": "Sheathing", "z": "Zone", "n": "None"},
    "spore-print-color": {"k": "Black", "n": "Brown", "b": "Buff", "h": "Chocolate", "g": "Green", "o": "Orange", "u": "Purple", "w": "White", "y": "Yellow"},
    "population": {"a": "Abundant", "c": "Clustered", "n": "Numerous", "s": "Scattered", "v": "Several", "y": "Solitary"},
    "habitat": {"g": "Grasses", "l": "Leaves", "m": "Meadows", "p": "Paths", "u": "Urban", "w": "Waste", "d": "Woods"},
}

# Collect user input using Streamlit widgets
user_input = {}

# Iterate through each feature and collect user input
for feature, options in feature_options.items():
    user_input[feature] = st.selectbox(f"{feature.replace('-', ' ').title()}:",
                                       list(options.values()))

# Display the user input
features = CustomData(
    cap_shape= user_input["cap-shape"],
    cap_surface=user_input["cap-surface"],
    cap_color=user_input["cap-color"],
    bruises=user_input["bruises"],
    odor=user_input["odor"],
    gill_attachment=user_input["gill-attachment"],
    gill_spacing=user_input["gill-spacing"],
    gill_size=user_input["gill-size"],
    gill_color=user_input["gill-color"],
    stalk_shape=user_input["stalk-shape"],
    stalk_root=user_input["stalk-root"],
    stalk_surface_above_ring=user_input["stalk-surface-above-ring"],
    stalk_surface_below_ring=user_input["stalk-surface-below-ring"],
    stalk_color_above_ring=user_input["stalk-color-above-ring"],
    stalk_color_below_ring=user_input["stalk-color-below-ring"],
    veil_type=user_input["veil-type"],
    veil_color=user_input["veil-color"],
    ring_number=user_input["ring-number"],
    ring_type=user_input["ring-type"],
    spore_print_color=user_input["spore-print-color"],
    population=user_input["population"],
    habitat=user_input["habitat"]
)

user_input_df = features.get_data_as_dataframe()

# Display the user input data
st.write("User Input Features:")
st.write(user_input_df)

# Add a Predict button
if st.button("Predict"):
    try:
        # Make predictions
        prediction = PredictPipeline().predict(features=user_input_df)

        # Display the raw prediction
        st.write("Raw Prediction:")
        st.write(prediction)

        # Interpret the prediction
        if prediction == "e":
            pred = "This Mushroom is Edible"
        else:
            pred = "This Mushroom is Poisonous!"

        # Display the final prediction
        st.write("Final Prediction:")
        st.write(pred)

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")