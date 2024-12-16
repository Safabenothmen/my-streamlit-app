import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_option_menu import option_menu

# Charger le modèle
model = pickle.load(open("diabetes_model3.pkl", "rb"))

# Charger le dataset
df = pd.read_csv("diabetes.csv")
df = df.drop(columns=["Insulin"])  # Supprimer la colonne Insulin

# Configuration de la page
st.set_page_config(page_title="Détection de Diabète", layout="wide", initial_sidebar_state="expanded")

# Barre de navigation avec style
with st.sidebar:
    selected = option_menu(
        "Menu Principal",
        ["Accueil", "Prédiction",],
        icons=["house", "activity",],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "#f0f2f6"},
            "icon": {"color": "#0000ff", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "center", "margin": "10px"},
            "nav-link-selected": {"background-color": "#00bfae", "color": "white"}
        }
    )

# 1. Page d'accueil
# 1. Page d'accueil
if selected == "Accueil":
    st.title("🩺 Détection de Diabète ")
    
    # Afficher l'image avec une taille ajustée
    
    # Texte d'accueil avec balises HTML et un style personnalisé
    st.markdown(
        """
        <div style="font-size: 18px; text-align: center; font-weight: 500; color: #333;">
            Bienvenue sur l'application de Détection de Diabète. Cette application utilise un modèle de machine learning pour prédire 
            la probabilité qu'un patient soit diabétique en se basant sur des données médicales telles que l'âge, l'indice de masse corporelle, 
            le taux de glucose, et bien d'autres informations. Suivez les étapes ci-dessous pour utiliser l'application efficacement.
        </div>
        """, unsafe_allow_html=True
    )
    
    # Guide étape par étape
    st.markdown(
        """
        ## 🚀 Comment utiliser l'application :
        
        1. **Page d'accueil** : 
            - Sur cette page, vous êtes accueillis avec un aperçu de l'application.
            - Lisez les instructions et familiarisez-vous avec le fonctionnement de l'application.

        2. **Prédiction du Diabète** : 
            - Allez dans la section **Prédiction** à partir du menu latéral.
            - Remplissez les informations médicales comme l'âge, le taux de glucose, l'indice de masse corporelle (BMI), etc.
            - Cliquez sur le bouton **Prédire** pour obtenir la prédiction de la probabilité que vous soyez diabétique.

  
        
        ## 📝 Astuce :
        - Assurez-vous de remplir tous les champs pour obtenir une prédiction plus précise.
        """, unsafe_allow_html=True
    )
    


# 2. Page de prédiction avec formulaire stylisé
elif selected == "Prédiction":
    st.header("Prédiction du Diabète")
    st.write("Veuillez entrer les informations suivantes pour prédire la probabilité de diabète.")

    # Fonction pour recueillir les entrées utilisateur avec un design amélioré
    def user_input_features():
        # Utilisation de "form" pour organiser les champs de saisie
        with st.form(key='user_input_form'):
            st.subheader("Informations médicales")
            pregnancies = st.number_input("Nombre de grossesses", min_value=0, max_value=20, value=1, step=1)
            glucose = st.number_input("Concentration de glucose (mg/dL)", min_value=0, max_value=200, value=120, step=5)
            blood_pressure = st.number_input("Pression sanguine (mm Hg)", min_value=0, max_value=122, value=70, step=1)
            skin_thickness = st.number_input("Épaisseur de la peau (mm)", min_value=0, max_value=100, value=20, step=1)
            bmi = st.number_input("Indice de masse corporelle (BMI)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
            dpf = st.number_input("Fonction d'hérédité du diabète (DPF)", min_value=0.0, max_value=3.0, value=0.5, step=0.1)
            age = st.number_input("Âge", min_value=1, max_value=120, value=25, step=1)

            # Ajouter un bouton de soumission dans le formulaire
            submit_button = st.form_submit_button(label="Prédire")

        # Si le bouton est cliqué, renvoyer les données saisies
        if submit_button:
            data = {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "BMI": bmi,
                "DiabetesPedigreeFunction": dpf,
                "Age": age,
            }
            return pd.DataFrame([data])
        else:
            return None

    user_data = user_input_features()

    # Afficher les données utilisateur dans un tableau
    if user_data is not None:
        st.subheader("Données saisies")
        st.write(user_data)  # Affiche les données sous forme de tableau

        # Prédiction avec un style plus dynamique
        prediction = model.predict(user_data)
        result = "Diabétique" if prediction[0] == 1 else "Non-Diabétique"
        st.subheader("Résultat de la Prédiction")
        st.success(f"Le patient est : **{result}**")
