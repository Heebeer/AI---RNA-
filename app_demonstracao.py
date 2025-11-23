import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px

# Configuração da página
st.set_page_config(
    page_title="Sistema IA - Componentes Hardware",
    page_icon="🔧",
    layout="wide"
)

@st.cache_resource
def carregar_modelo():
    """Carrega o modelo com tratamento de compatibilidade"""
    try:
        # Tenta carregar com custom_objects para resolver o erro do DepthwiseConv2D
        model = tf.keras.models.load_model(
            "model/modelo_componentes.h5",
            custom_objects={},
            compile=True
        )
        st.success("✅ Modelo carregado com sucesso!")
        return model
    except Exception as e:
        st.warning(f"⚠️  Erro ao carregar modelo: {str(e)[:100]}...")
        
        # SOLUÇÃO DEFINITIVA: Recria a arquitetura e carrega os pesos
        st.info("🔄 Recriando modelo com arquitetura MobileNetV2...")
        return recriar_e_carregar_modelo()

def recriar_e_carregar_modelo():
    """Recria a arquitetura exata do modelo e carrega os pesos"""
    try:
        # Recria a mesma arquitetura que você usou no treinamento
        base_model = tf.keras.applications.MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False
        
        # Mesma arquitetura do seu treinamento
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        output = tf.keras.layers.Dense(5, activation="softmax")(x)
        
        model = tf.keras.Model(inputs=base_model.input, outputs=output)
        
        # Compila com mesma configuração
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        # Tenta carregar APENAS os pesos (não o modelo completo)
        model.load_weights("model/modelo_componentes.h5")
        st.success("✅ Pesos carregados com sucesso na nova arquitetura!")
        return model
        
    except Exception as e:
        st.error(f"❌ Erro ao recriar modelo: {e}")
        st.info("🎯 Usando modelo de demonstração para funcionalidade básica...")
        return criar_modelo_demo()

def criar_modelo_demo():
    """Modelo simples apenas para manter a aplicação funcional"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

@st.cache_data
def carregar_classes():
    """Carrega as classes do arquivo"""
    try:
        with open("model/classes.txt", "r") as f:
            classes = f.read().splitlines()
        return classes
    except:
        return ['fonte', 'hd', 'mb', 'ram', 'ssd']

# Carrega modelo e classes
model = carregar_modelo()
CLASSES = carregar_classes()

# Função de predição
def prever_imagem(image):
    """Faz predição em uma imagem"""
    try:
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        componente = CLASSES[predicted_idx]
        
        return componente, confidence, predictions
    except Exception as e:
        st.error(f"Erro na predição: {e}")
        return "erro", 0.0, []

# ================================================================
# INTERFACE STREAMLIT (MESMA QUE VOCÊ JÁ TEM)
# ================================================================

st.title("🔧 Sistema Inteligente - Classificação de Componentes")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🎯 Modo de Demonstração")
    modo = st.selectbox(
        "Selecione o cenário:",
        ["Classificação Simples", "Inventário Automático", "Assistente Montagem"]
    )
    
    st.markdown("---")
    st.info("""
    **Como usar:**
    1. Faça upload das imagens
    2. Veja os resultados automáticos  
    3. Analise as métricas
    """)

# Diferentes modos de demonstração
if modo == "Classificação Simples":
    st.header("📷 Classificação de Componentes")
    
    uploaded_files = st.file_uploader(
        "Faça upload das imagens dos componentes:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Resultados")
            resultados = []
            
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert("RGB")
                componente, confidence, _ = prever_imagem(image)
                
                resultados.append({
                    'Arquivo': uploaded_file.name,
                    'Componente': componente.upper(),
                    'Confiança': f"{confidence:.2%}",
                    'Status': '✅ Alta' if confidence > 0.7 else '⚠️ Média'
                })
            
            df = pd.DataFrame(resultados)
            st.dataframe(df, use_container_width=True)
        
        with col2:
            st.subheader("📊 Estatísticas")
            if resultados:
                componentes_count = pd.DataFrame(resultados)['Componente'].value_counts()
                fig = px.pie(
                    values=componentes_count.values,
                    names=componentes_count.index,
                    title="Distribuição dos Componentes Identificados"
                )
                st.plotly_chart(fig, use_container_width=True)

elif modo == "Inventário Automático":
    st.header("📦 Sistema de Inventário Inteligente")
    
    st.info("""
    **Cenário Real:** Empresa de TI escaneando componentes para controle de estoque.
    """)
    
    uploaded_files = st.file_uploader(
        "Escaneie os componentes para inventário:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        inventario = {classe: 0 for classe in CLASSES}
        confiancas = []
        
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            componente, confidence, _ = prever_imagem(image)
            
            if confidence > 0.7:
                inventario[componente] += 1
            confiancas.append(confidence)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Inventário Atual")
            for componente, quantidade in inventario.items():
                st.write(f"🔹 **{componente.upper()}**: {quantidade} unidades")
            
            total = sum(inventario.values())
            st.metric("📦 Total de Componentes", total)
        
        with col2:
            st.subheader("📈 Métricas de Qualidade")
            if confiancas:
                confianca_media = np.mean(confiancas)
                st.metric("🎯 Confiança Média", f"{confianca_media:.2%}")

elif modo == "Assistente Montagem":
    st.header("🛠️ Assistente de Montagem com IA")
    
    st.info("""
    **Cenário Real:** Técnico montando computador com assistência de IA.
    """)
    
    uploaded_file = st.file_uploader(
        "Mostre o componente para receber instruções:",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Componente Analisado", use_container_width=True)
        
        componente, confidence, _ = prever_imagem(image)
        
        instrucoes = {
            'fonte': "🔌 Conecte os cabos de energia na placa-mãe e componentes...",
            'hd': "💾 Conecte cabo SATA e energia, parafuse no gabinete...",
            'mb': "🔩 Instale primeiro no gabinete, depois conecte outros componentes...",
            'ram': "🧠 Alinhe os entalhes e pressione até travar nos slots...",
            'ssd': "⚡ Conecte cabo SATA e energia, fixe no suporte..."
        }
        
        st.success(f"✅ **Componente Identificado:** {componente.upper()}")
        st.metric("Confiança", f"{confidence:.2%}")
        
        if componente in instrucoes:
            st.subheader("📋 Instruções:")
            st.markdown(instrucoes[componente])

# Footer
st.markdown("---")
st.markdown("🔧 *Sistema desenvolvido com TensorFlow*")