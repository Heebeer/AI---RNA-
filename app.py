import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Carregar modelo
model = tf.keras.models.load_model("model/modelo_componentes.h5")

# Ler classes do arquivo
with open("model/classes.txt") as f:
    CLASSES = f.read().splitlines()

st.title("Classificação de Componentes de Hardware")

uploaded_file = st.file_uploader("Envie uma imagem...", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)

    pred = model.predict(img_array)[0]
    index = np.argmax(pred)
    confidence = pred[index]

    st.subheader(f"Componente detectado: **{CLASSES[index]}**")
    st.write(f"Confiança: {confidence*100:.2f}%")
