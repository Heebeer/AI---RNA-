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
        # Tenta carregar normalmente
        model = tf.keras.models.load_model("model/modelo_componentes.h5")
        st.success("✅ Modelo carregado com sucesso!")
        return model
    except Exception as e:
        st.warning(f"⚠️  Erro ao carregar modelo: {e}")
        st.info("🔄 Usando modelo de demonstração...")
        
        # Cria um modelo simples para demonstração
        return criar_modelo_demo()

def criar_modelo_demo():
    """Cria um modelo simples para demonstração quando o original falha"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    # Compila o modelo (não treinado, apenas para demo)
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
        # Fallback se o arquivo não existir
        return ['fonte', 'hd', 'mb', 'ram', 'ssd']

# Carrega modelo e classes
model = carregar_modelo()
CLASSES = carregar_classes()

# Interface principal
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

# Função de predição
def prever_imagem(image):
    """Faz predição em uma imagem"""
    try:
        # Pré-processamento
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predição
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        componente = CLASSES[predicted_idx]
        
        return componente, confidence, predictions
    except Exception as e:
        st.error(f"Erro na predição: {e}")
        return "erro", 0.0, []

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
                # Processa imagem
                image = Image.open(uploaded_file).convert("RGB")
                
                # Predição
                componente, confidence, _ = prever_imagem(image)
                
                resultados.append({
                    'Arquivo': uploaded_file.name,
                    'Componente': componente.upper(),
                    'Confiança': f"{confidence:.2%}",
                    'Status': '✅ Alta' if confidence > 0.7 else '⚠️ Média'
                })
            
            # Tabela de resultados
            df = pd.DataFrame(resultados)
            st.dataframe(df, use_container_width=True)
        
        with col2:
            st.subheader("📊 Estatísticas")
            
            # Gráfico de distribuição
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
    O sistema identifica automaticamente e atualiza o inventário.
    """)
    
    uploaded_files = st.file_uploader(
        "Escaneie os componentes para inventário:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Simula inventário
        inventario = {classe: 0 for classe in CLASSES}
        confiancas = []
        
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            componente, confidence, _ = prever_imagem(image)
            
            if confidence > 0.7:
                inventario[componente] += 1
            confiancas.append(confidence)
        
        # Mostra inventário
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
                
                if confianca_media > 0.8:
                    st.success("✅ Excelente qualidade nas identificações!")
                elif confianca_media > 0.6:
                    st.warning("⚠️  Qualidade aceitável")
                else:
                    st.error("❌ Qualidade baixa - verifique as imagens")

elif modo == "Assistente Montagem":
    st.header("🛠️ Assistente de Montagem com IA")
    
    st.info("""
    **Cenário Real:** Técnico montando computador - a IA identifica cada componente 
    e fornece instruções específicas de instalação.
    """)
    
    uploaded_file = st.file_uploader(
        "Mostre o componente para receber instruções:",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file:
        # Processa imagem
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Componente Analisado", use_container_width=True)
        
        # Predição
        componente, confidence, _ = prever_imagem(image)
        
        # Instruções específicas
        instrucoes = {
            'fonte': """
            **🔌 Instalação da Fonte de Alimentação:**
            1. Posicione a fonte no gabinete
            2. Parafuse firmemente
            3. Conecte o cabo de 24 pinos na placa-mãe
            4. Conecte o cabo de 4/8 pinos CPU
            5. Conecte os cabos SATA/PCIe nos componentes
            """,
            'hd': """
            **💾 Instalação do HD:**
            1. Encaixe no bay 3.5" do gabinete
            2. Parafuse dos dois lados
            3. Conecte cabo SATA na placa-mãe
            4. Conecte cabo de energia da fonte
            """,
            'mb': """
            **🔩 Instalação da Placa-Mãe:**
            1. Instale os standoffs no gabinete
            2. Posicione a placa-mãe
            3. Parafuse todos os pontos
            4. Conecte painel frontal
            5. Conecte alimentação 24-pin + CPU
            """,
            'ram': """
            **🧠 Instalação da Memória RAM:**
            1. Abra as travas dos slots
            2. Alinhe o entalhe da RAM com o slot
            3. Pressione firmemente até travar
            4. Ouça o 'click' de encaixe
            """,
            'ssd': """
            **⚡ Instalação do SSD:**
            1. Encaixe no bay 2.5" do gabinete
            2. Parafuse ou use sistema tool-less
            3. Conecte cabo SATA na placa-mãe
            4. Conecte cabo de energia da fonte
            """
        }
        
        st.success(f"✅ **Componente Identificado:** {componente.upper()}")
        st.metric("Confiança da Identificação", f"{confidence:.2%}")
        
        if componente in instrucoes:
            st.subheader("📋 Instruções de Montagem:")
            st.markdown(instrucoes[componente])
        else:
            st.warning("Instruções não disponíveis para este componente")

# Footer
st.markdown("---")
st.markdown(
    "🔧 *Sistema desenvolvido com TensorFlow - Classificação de Componentes de Hardware*"
)