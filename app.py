import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# Configuração da página
st.set_page_config(
    page_title="Classificador de Hardware",
    page_icon="🔧",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Carrega o modelo uma única vez"""
    try:
        model = tf.keras.models.load_model("model/modelo_componentes.h5")
        return model
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None

@st.cache_data
def load_classes():
    """Carrega as classes do arquivo"""
    try:
        with open("model/classes.txt", "r") as f:
            classes = f.read().splitlines()
        return classes
    except Exception as e:
        st.error(f"Erro ao carregar classes: {e}")
        return []

def preprocess_image(image):
    """Pré-processa a imagem para o modelo"""
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Interface principal
st.title("🔧 Classificador de Componentes de Hardware")
st.markdown("---")

# Sidebar com informações
with st.sidebar:
    st.header("ℹ️ Informações")
    st.write("**Componentes suportados:**")
    
    # Carrega classes
    CLASSES = load_classes()
    for class_name in CLASSES:
        st.write(f"- {class_name.upper()}")
    
    st.markdown("---")
    st.write("**Instruções:**")
    st.write("1. Faça upload de uma imagem")
    st.write("2. Aguarde a classificação")
    st.write("3. Veja o resultado e confiança")

# Upload de imagem
uploaded_file = st.file_uploader(
    "📤 Faça upload de uma imagem do componente", 
    type=["jpg", "jpeg", "png"],
    help="Formatos suportados: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    # Carrega e exibe a imagem
    try:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Imagem Original")
            st.image(image, use_container_width=True)
        
        with col2:
            # Pré-processamento e predição
            with st.spinner("🔍 Analisando imagem..."):
                model = load_model()
                if model is not None:
                    processed_image = preprocess_image(image)
                    
                    # Simula um tempo de processamento para melhor UX
                    time.sleep(0.5)
                    
                    predictions = model.predict(processed_image, verbose=0)[0]
                    predicted_class_idx = np.argmax(predictions)
                    confidence = predictions[predicted_class_idx]
                    
                    # Resultado
                    st.subheader("📊 Resultado")
                    
                    if CLASSES:
                        predicted_class = CLASSES[predicted_class_idx]
                        
                        # Barra de confiança
                        st.metric(
                            label="**Componente Identificado**",
                            value=predicted_class.upper()
                        )
                        
                        st.metric(
                            label="**Confiança**",
                            value=f"{confidence*100:.2f}%"
                        )
                        
                        # Barra visual de confiança
                        st.progress(float(confidence))
                        
                        # Alertas baseados na confiança
                        if confidence > 0.8:
                            st.success("✅ Alta confiança na predição!")
                        elif confidence > 0.5:
                            st.warning("⚠️ Confiança moderada na predição")
                        else:
                            st.error("❌ Baixa confiança - considere verificar a imagem")
                        
                        # Mostra todas as probabilidades
                        st.subheader("📈 Todas as Probabilidades")
                        for i, (class_name, prob) in enumerate(zip(CLASSES, predictions)):
                            color = "green" if i == predicted_class_idx else "gray"
                            st.write(
                                f"<span style='color: {color}; font-weight: {'bold' if i == predicted_class_idx else 'normal'};'>"
                                f"{class_name}: {prob*100:.2f}%</span>",
                                unsafe_allow_html=True
                            )
                    else:
                        st.error("Não foi possível carregar as classes do modelo")
                
    except Exception as e:
        st.error(f"Erro ao processar imagem: {e}")

else:
    # Estado inicial
    st.info("👆 Faça upload de uma imagem para começar a classificação")
    
    # Exemplo de imagens esperadas
    st.markdown("---")
    st.subheader("💡 Exemplos de componentes:")
    st.write("""
    - **Fonte de alimentação**
    - **HD (Disco Rígido)**
    - **MB (Placa-mãe)**
    - **RAM (Memória)**
    - **SSD (Unidade de Estado Sólido)**
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Classificador de Componentes de Hardware • Desenvolvido com TensorFlow e Streamlit"
    "</div>",
    unsafe_allow_html=True
)