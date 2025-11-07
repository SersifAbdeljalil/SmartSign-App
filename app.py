import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
from pathlib import Path

# ============================================
# CONFIGURATION
# ============================================

st.set_page_config(
    page_title="ASL Gesture Recognition",
    page_icon="🤟",
    layout="wide"
)

# Chemins des fichiers
MODEL_DIR = r"C:\Users\sersi\Desktop\projet_SE_et_IOT\HandSense_project\model_mobile"

# ============================================
# FONCTION: CHARGER LE MODÈLE
# ============================================

@st.cache_resource
def load_model():
    """Charge le modèle, scaler et encoder"""
    try:
        model_path = os.path.join(MODEL_DIR, "asl_model_mobile.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        
        return model, scaler, label_encoder, None
    except Exception as e:
        return None, None, None, str(e)

# ============================================
# FONCTION: EMOJI POUR CHAQUE GESTE
# ============================================

def get_emoji(gesture):
    """Retourne un emoji pour chaque geste"""
    emoji_map = {
        'A': '✊', 'B': '🤚', 'C': '👌', 'D': '☝️', 'E': '✋',
        'F': '👌', 'G': '👈', 'H': '✌️', 'I': '🤙', 'J': '🤙',
        'K': '✌️', 'L': '👆', 'M': '✊', 'N': '✊', 'O': '👌',
        'P': '👇', 'Q': '👇', 'R': '🤞', 'S': '✊', 'T': '👊',
        'U': '✌️', 'V': '✌️', 'W': '🤟', 'X': '☝️', 'Y': '🤙',
        'Z': '☝️',
        'Hello': '👋', 'Thanks': '🙏', 'Yes': '👍', 'No': '👎',
        'Please': '🙏', 'Sorry': '🤷', 'Help': '🆘'
    }
    return emoji_map.get(gesture, '🤟')

# ============================================
# INTERFACE PRINCIPALE
# ============================================

def main():
    # En-tête
    st.title("🤟 ASL Gesture Recognition")
    st.markdown("### Testez votre modèle de reconnaissance de gestes ASL")
    
    # Charger le modèle
    model, scaler, label_encoder, error = load_model()
    
    if error:
        st.error(f"❌ Erreur de chargement du modèle: {error}")
        st.info("📁 Vérifiez que les fichiers sont dans: " + MODEL_DIR)
        return
    
    st.success("✅ Modèle chargé avec succès!")
    
    # Sidebar: Informations
    with st.sidebar:
        st.header("📊 Informations du Modèle")
        st.metric("Classes", len(label_encoder.classes_))
        st.metric("Features", 11)
        
        st.markdown("---")
        st.markdown("### 🎯 Classes reconnues")
        cols = st.columns(2)
        for i, classe in enumerate(label_encoder.classes_):
            with cols[i % 2]:
                st.markdown(f"{get_emoji(classe)} **{classe}**")
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["🎮 Test Manuel", "📁 Upload CSV", "📊 Exemples"])
    
    # ============================================
    # TAB 1: TEST MANUEL
    # ============================================
    with tab1:
        st.header("🎮 Test avec valeurs manuelles")
        st.markdown("Ajustez les valeurs des capteurs ci-dessous:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🖐️ Flex Sensors")
            flex_1 = st.slider("Flex 1 (Pouce)", 0, 100, 50)
            flex_2 = st.slider("Flex 2 (Index)", 0, 100, 50)
            flex_3 = st.slider("Flex 3 (Majeur)", 0, 100, 50)
            flex_4 = st.slider("Flex 4 (Annulaire)", 0, 100, 50)
            flex_5 = st.slider("Flex 5 (Auriculaire)", 0, 100, 50)
        
        with col2:
            st.subheader("🔄 Gyroscope (rad/s)")
            gyr_x = st.slider("GYR X", -1.0, 1.0, 0.0, 0.1)
            gyr_y = st.slider("GYR Y", -1.0, 1.0, 0.0, 0.1)
            gyr_z = st.slider("GYR Z", -1.0, 1.0, 0.0, 0.1)
        
        with col3:
            st.subheader("📐 Accéléromètre (m/s²)")
            acc_x = st.slider("ACC X", -10.0, 10.0, 0.0, 0.5)
            acc_y = st.slider("ACC Y", -10.0, 10.0, 0.0, 0.5)
            acc_z = st.slider("ACC Z", -10.0, 10.0, 9.8, 0.5)
        
        st.markdown("---")
        
        # Bouton de prédiction
        if st.button("🚀 PRÉDIRE LE GESTE", type="primary", use_container_width=True):
            # Préparer les données
            features = np.array([[flex_1, flex_2, flex_3, flex_4, flex_5,
                                  gyr_x, gyr_y, gyr_z,
                                  acc_x, acc_y, acc_z]])
            
            # Normaliser
            features_scaled = scaler.transform(features)
            
            # Prédire
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            predicted_class = label_encoder.classes_[prediction]
            confidence = probabilities[prediction]
            
            # Affichage des résultats
            st.markdown("---")
            st.markdown("## 🎯 RÉSULTAT")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"# {get_emoji(predicted_class)} **{predicted_class}**")
            
            with col2:
                st.metric("Confiance", f"{confidence*100:.1f}%")
            
            with col3:
                if confidence > 0.9:
                    st.success("Très confiant ✅")
                elif confidence > 0.7:
                    st.warning("Confiant ⚠️")
                else:
                    st.error("Faible 🔻")
            
            # Graphique des probabilités (Top 5)
            st.markdown("### 📊 Top 5 Prédictions")
            top_5_idx = np.argsort(probabilities)[-5:][::-1]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=[probabilities[i]*100 for i in top_5_idx],
                    y=[label_encoder.classes_[i] for i in top_5_idx],
                    orientation='h',
                    text=[f"{probabilities[i]*100:.1f}%" for i in top_5_idx],
                    textposition='auto',
                    marker=dict(
                        color=['#FF6B6B' if i == prediction else '#4ECDC4' 
                               for i in top_5_idx]
                    )
                )
            ])
            
            fig.update_layout(
                height=300,
                xaxis_title="Probabilité (%)",
                yaxis_title="Classe",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # TAB 2: UPLOAD CSV
    # ============================================
    with tab2:
        st.header("📁 Upload un fichier CSV")
        st.markdown("Le CSV doit contenir les 11 colonnes de features:")
        st.code("flex_1, flex_2, flex_3, flex_4, flex_5, GYRx, GYRy, GYRz, ACCx, ACCy, ACCz")
        
        uploaded_file = st.file_uploader("Choisir un fichier CSV", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Fichier chargé: {df.shape[0]} lignes")
                
                # Vérifier les colonnes
                required_cols = ['flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5',
                                'GYRx', 'GYRy', 'GYRz', 'ACCx', 'ACCy', 'ACCz']
                
                if not all(col in df.columns for col in required_cols):
                    st.error("❌ Colonnes manquantes dans le CSV!")
                    st.info("Colonnes requises: " + ", ".join(required_cols))
                    return
                
                # Afficher un aperçu
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("🚀 PRÉDIRE TOUT LE FICHIER", type="primary"):
                    with st.spinner("Prédiction en cours..."):
                        # Extraire les features
                        X = df[required_cols].values
                        
                        # Normaliser
                        X_scaled = scaler.transform(X)
                        
                        # Prédire
                        predictions = model.predict(X_scaled)
                        probabilities = model.predict_proba(X_scaled)
                        
                        # Ajouter au dataframe
                        df['prediction'] = [label_encoder.classes_[p] for p in predictions]
                        df['confidence'] = [probabilities[i][predictions[i]] for i in range(len(predictions))]
                        df['emoji'] = df['prediction'].apply(get_emoji)
                    
                    st.success("✅ Prédictions terminées!")
                    
                    # Résultats
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total prédictions", len(df))
                    
                    with col2:
                        st.metric("Confiance moyenne", f"{df['confidence'].mean()*100:.1f}%")
                    
                    # Afficher les résultats
                    st.dataframe(
                        df[['emoji', 'prediction', 'confidence'] + required_cols],
                        use_container_width=True
                    )
                    
                    # Télécharger les résultats
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Télécharger les résultats",
                        csv,
                        "predictions.csv",
                        "text/csv"
                    )
                    
                    # Distribution des prédictions
                    st.markdown("### 📊 Distribution des prédictions")
                    pred_counts = df['prediction'].value_counts()
                    
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=pred_counts.index,
                            values=pred_counts.values,
                            hole=0.3
                        )
                    ])
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Erreur: {e}")
    
    # ============================================
    # TAB 3: EXEMPLES PRÉ-DÉFINIS
    # ============================================
    with tab3:
        st.header("📊 Exemples pré-définis")
        st.markdown("Testez avec des exemples types de gestes")
        
        examples = {
            "Position repos": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.8],
            "Main ouverte": [10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 9.8],
            "Poing fermé": [90, 90, 90, 90, 90, 0, 0, 0, 0, 0, 9.8],
            "Pointage index": [90, 10, 90, 90, 90, 0, 0, 0, 0, 0, 9.8],
            "Signe de paix": [90, 10, 10, 90, 90, 0, 0, 0, 0, 0, 9.8],
        }
        
        selected_example = st.selectbox("Choisir un exemple", list(examples.keys()))
        
        if st.button("🎯 TESTER CET EXEMPLE", type="primary"):
            features = np.array([examples[selected_example]])
            features_scaled = scaler.transform(features)
            
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            predicted_class = label_encoder.classes_[prediction]
            confidence = probabilities[prediction]
            
            # Résultats
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"## {get_emoji(predicted_class)} **{predicted_class}**")
            
            with col2:
                st.metric("Confiance", f"{confidence*100:.1f}%")
            
            # Top 3
            st.markdown("### Top 3 prédictions")
            top_3_idx = np.argsort(probabilities)[-3:][::-1]
            
            for i, idx in enumerate(top_3_idx, 1):
                classe = label_encoder.classes_[idx]
                prob = probabilities[idx]
                st.markdown(f"{i}. {get_emoji(classe)} **{classe}** - {prob*100:.1f}%")

# ============================================
# LANCEMENT
# ============================================

if __name__ == "__main__":
    main()