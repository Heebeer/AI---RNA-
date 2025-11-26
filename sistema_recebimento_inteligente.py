import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Configuração da página
st.set_page_config(
    page_title="Sistema de Recebimento - TechLog Solutions",
    page_icon="🏢",
    layout="wide"
)

# ================================================================
# CONFIGURAÇÕES DO SISTEMA ATUALIZADO
# ================================================================

@st.cache_resource
def carregar_modelo_atualizado():
    """Carrega o modelo atualizado"""
    try:
        model = tf.keras.models.load_model("model/modelo_atualizado.h5")
        return model
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {e}")
        return None

@st.cache_data
def carregar_classes_atualizadas():
    """Carrega as classes do arquivo"""
    try:
        with open("model/classes.txt", "r") as f:
            return f.read().splitlines()
    except:
        # Fallback: retorna classes padrão
        return ['fonte', 'hd', 'mb', 'processador', 'ssd']

# ================================================================
# CONFIGURAÇÃO DINÂMICA BASEADA NAS CLASSES
# ================================================================

def carregar_configuracoes_dinamicas():
    """Carrega configurações baseadas nas classes disponíveis"""
    classes = carregar_classes_atualizadas()
    
    # Mapeamento de localização no armazém (dinâmico)
    LOCALIZACOES = {}
    PRECOS = {}
    EMOJIS = {}
    
    # Configurações padrão para cada classe possível
    config_padrao = {
        'fonte': {'setor': 'A', 'corredor': '01', 'prateleira': 'A1-A5', 'zona': 'Energia', 'preco': 150.00, 'emoji': '🔌'},
        'hd': {'setor': 'B', 'corredor': '02', 'prateleira': 'B1-B8', 'zona': 'Armazenamento', 'preco': 80.00, 'emoji': '💾'},
        'mb': {'setor': 'C', 'corredor': '03', 'prateleira': 'C1-C6', 'zona': 'Processamento', 'preco': 500.00, 'emoji': '🔌'},
        'processador': {'setor': 'D', 'corredor': '04', 'prateleira': 'D1-D4', 'zona': 'CPU', 'preco': 800.00, 'emoji': '⚡'},
        'ssd': {'setor': 'F', 'corredor': '06', 'prateleira': 'F1-F7', 'zona': 'Armazenamento Rápido', 'preco': 200.00, 'emoji': '🚀'}
    }
    
    # Aplica apenas para as classes que existem
    for classe in classes:
        if classe in config_padrao:
            config = config_padrao[classe]
            LOCALIZACOES[classe] = {
                'setor': config['setor'],
                'corredor': config['corredor'],
                'prateleira': config['prateleira'],
                'zona': config['zona'],
                'instrucoes': f'Armazenar {classe} conforme procedimento padrão'
            }
            PRECOS[classe] = config['preco']
            EMOJIS[classe] = config['emoji']
        else:
            # Configuração padrão para classes não previstas
            LOCALIZACOES[classe] = {
                'setor': 'Z',
                'corredor': '99',
                'prateleira': 'Z1-Z3',
                'zona': 'Geral',
                'instrucoes': f'Armazenar {classe} em área geral'
            }
            PRECOS[classe] = 100.00
            EMOJIS[classe] = '🔧'
    
    return LOCALIZACOES, PRECOS, EMOJIS

# ================================================================
# SISTEMA DE RECEBIMENTO
# ================================================================

class SistemaRecebimentoAtualizado:
    def __init__(self):
        self.model = carregar_modelo_atualizado()
        self.classes = carregar_classes_atualizadas()
        self.LOCALIZACOES, self.PRECOS, self.EMOJIS = carregar_configuracoes_dinamicas()
        
        # Inicializar inventário
        self.inventario = {classe: 0 for classe in self.classes}
        self.historico = []
        self.lote_atual = []
        
        # Carregar dados persistentes se existirem
        self._carregar_dados_persistentes()
    
    def _carregar_dados_persistentes(self):
        """Carrega dados do session_state se existirem"""
        if 'inventario_atualizado' in st.session_state:
            self.inventario = st.session_state.inventario_atualizado
        if 'historico_atualizado' in st.session_state:
            self.historico = st.session_state.historico_atualizado
    
    def _salvar_dados_persistentes(self):
        """Salva dados no session_state"""
        st.session_state.inventario_atualizado = self.inventario
        st.session_state.historico_atualizado = self.historico
    
    def processar_componente(self, image, numero_serie="N/A", fornecedor="N/A"):
        """Processa um componente individual"""
        if self.model is None:
            st.error("❌ Modelo não carregado corretamente")
            return None
        
        # Pré-processamento
        image_resized = image.resize((224, 224))
        img_array = np.array(image_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predição
        predictions = self.model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        componente = self.classes[predicted_idx]
        
        # Registro do componente
        registro = {
            'timestamp': datetime.now(),
            'componente': componente,
            'confianca': float(confidence),
            'numero_serie': numero_serie,
            'fornecedor': fornecedor,
            'localizacao': self.LOCALIZACOES[componente],
            'preco': self.PRECOS[componente],
            'emoji': self.EMOJIS[componente]
        }
        
        self.inventario[componente] += 1
        self.lote_atual.append(registro)
        self.historico.append(registro)
        
        # Salva dados persistentes
        self._salvar_dados_persistentes()
        
        return registro
    
    def finalizar_lote(self, numero_lote):
        """Finaliza o lote atual e gera relatório"""
        if not self.lote_atual:
            return None
        
        total_componentes = len(self.lote_atual)
        valor_total = sum(item['preco'] for item in self.lote_atual)
        
        relatorio = {
            'numero_lote': numero_lote,
            'timestamp': datetime.now(),
            'total_componentes': total_componentes,
            'valor_total': valor_total,
            'itens': self.lote_atual.copy()
        }
        
        self.lote_atual.clear()
        self._salvar_dados_persistentes()
        
        return relatorio

# ================================================================
# INICIALIZAÇÃO DO SISTEMA
# ================================================================

if 'sistema_atualizado' not in st.session_state:
    st.session_state.sistema_atualizado = SistemaRecebimentoAtualizado()

if 'lote_counter_atualizado' not in st.session_state:
    st.session_state.lote_counter_atualizado = 1

if 'uploaded_files_cache_atualizado' not in st.session_state:
    st.session_state.uploaded_files_cache_atualizado = []

if 'lote_processado_atualizado' not in st.session_state:
    st.session_state.lote_processado_atualizado = False

sistema = st.session_state.sistema_atualizado

# ================================================================
# INTERFACE STREAMLIT
# ================================================================

st.title("🏢 TechLog Solutions")
st.markdown(f"**🔧 Componentes: {', '.join([f'{sistema.EMOJIS[c]} {c.upper()}' for c in sistema.classes])}**")
st.markdown("---")

# Sidebar atualizada
with st.sidebar:
    st.header("🏭 Controle de Operações")
    
    st.subheader("📊 Estatísticas")
    total_componentes = sum(sistema.inventario.values())
    st.metric("Componentes Recebidos", total_componentes)
    st.metric("Lotes Processados", st.session_state.lote_counter_atualizado - 1)
    
    st.subheader("🔧 Inventário Atual")
    for componente, quantidade in sistema.inventario.items():
        emoji = sistema.EMOJIS.get(componente, '🔧')
        st.write(f"{emoji} {componente.upper()}: {quantidade} un")
    
    st.markdown("---")
    st.info(f"""
    **Componentes Reconhecidos:**
    {' | '.join([f'{sistema.EMOJIS[c]} {c.upper()}' for c in sistema.classes])}
    """)

# Abas principais
tab1, tab2, tab3 = st.tabs(["📦 Recebimento", "🗺️ Mapa do Armazém", "📊 Dashboard"])

# ABA 1: RECEBIMENTO ATUALIZADO
with tab1:
    st.header("📦 Área de Recebimento - Sistema Atualizado")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("1. Upload do Lote")
        
        uploaded_files = st.file_uploader(
            "Selecione as imagens dos componentes:",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"uploader_atualizado_{st.session_state.lote_counter_atualizado}"
        )
        
        if uploaded_files and not st.session_state.lote_processado_atualizado:
            st.session_state.uploaded_files_cache_atualizado = uploaded_files
        
        current_files = st.session_state.uploaded_files_cache_atualizado
        if current_files and not st.session_state.lote_processado_atualizado:
            st.success(f"📁 **Lote carregado:** {len(current_files)} componente(s)")
            
            # Grid de preview
            cols = st.columns(4)
            for i, file in enumerate(current_files):
                with cols[i % 4]:
                    img = Image.open(file)
                    st.image(img, width=80, caption=f"Item {i+1}")
        
        st.subheader("2. Informações do Lote")
        with st.form("lote_form_atualizado"):
            numero_lote = st.text_input("Número do Lote", f"LOTE-{st.session_state.lote_counter_atualizado:03d}")
            fornecedor = st.selectbox("Fornecedor", ["Intel", "AMD", "Dell", "HP", "Lenovo", "Asus", "Samsung", "Kingston", "Seagate", "Western Digital", "Outro"])
            responsavel = st.text_input("Responsável pelo Recebimento", "Operador 01")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                processar_btn = st.form_submit_button("🚀 Processar Lote", type="primary", 
                                                    disabled=not current_files or st.session_state.lote_processado_atualizado)
            with col_btn2:
                limpar_btn = st.form_submit_button("🗑️ Limpar Lote")
        
        # Processamento do lote
        if processar_btn and current_files and not st.session_state.lote_processado_atualizado:
            with st.spinner(f"🔍 Processando {len(current_files)} componentes..."):
                resultados = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(current_files):
                    status_text.text(f"Processando item {i+1} de {len(current_files)}...")
                    
                    try:
                        image = Image.open(file).convert("RGB")
                        n_serie = f"TL{st.session_state.lote_counter_atualizado:03d}-{i+1:03d}"
                        
                        resultado = sistema.processar_componente(image, n_serie, fornecedor)
                        if resultado:
                            resultados.append(resultado)
                    
                    except Exception as e:
                        st.error(f"Erro ao processar item {i+1}: {e}")
                    
                    progress_bar.progress((i + 1) / len(current_files))
                    time.sleep(0.2)
                
                if resultados:
                    relatorio = sistema.finalizar_lote(numero_lote)
                    st.session_state.lote_counter_atualizado += 1
                    st.session_state.lote_processado_atualizado = True
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.balloons()
                    st.success(f"✅ **Lote {numero_lote} processado com sucesso!**")
                    st.success(f"**{len(resultados)}** itens adicionados ao inventário.")
                    
                    # Mostra detalhes do lote processado
                    with st.expander("📄 Ver detalhes do lote processado", expanded=True):
                        for i, item in enumerate(resultados, 1):
                            cor = "🟢" if item['confianca'] > 0.8 else "🟡" if item['confianca'] > 0.6 else "🔴"
                            st.write(f"{cor} **Item {i}:** {item['emoji']} {item['componente'].upper()} - {item['numero_serie']} "
                                   f"(Confiança: {item['confianca']:.1%})")
                else:
                    st.error("❌ Nenhum item foi processado corretamente")
        
        # Limpar lote
        if limpar_btn:
            st.session_state.uploaded_files_cache_atualizado = []
            st.session_state.lote_processado_atualizado = False
            st.rerun()
        
        # Botão para novo lote
        if st.session_state.lote_processado_atualizado:
            st.markdown("---")
            if st.button("🔄 Iniciar Novo Lote", type="primary"):
                st.session_state.uploaded_files_cache_atualizado = []
                st.session_state.lote_processado_atualizado = False
                st.rerun()
    
    with col2:
        st.subheader("📊 Estoque Atual")
        for componente, quantidade in sistema.inventario.items():
            emoji = sistema.EMOJIS.get(componente, '🔧')
            st.metric(
                label=f"{emoji} {componente.upper()}",
                value=f"{quantidade} un",
                delta=None
            )
        
        st.markdown("---")
        st.subheader("🏷️ Últimas Etiquetas")
        
        if sistema.historico:
            ultimos_itens = sistema.historico[-3:]
            for item in reversed(ultimos_itens):
                loc = item['localizacao']
                cor_confianca = "🟢" if item['confianca'] > 0.8 else "🟡" if item['confianca'] > 0.6 else "🔴"
                
                with st.container():
                    st.markdown(f"""
                    **{item['emoji']} {item['componente'].upper()}** - {item['numero_serie']}
                    {cor_confianca} **Confiança:** {item['confianca']:.1%}
                    📍 **Local:** Setor {loc['setor']}, C{loc['corredor']}
                    🏷️ **Prateleira:** {loc['prateleira']}
                    """)
                    st.markdown("---")
        else:
            st.info("📦 Nenhum componente processado ainda")

# ABA 2: MAPA DO ARMAZÉM ATUALIZADO
with tab2:
    st.header("🗺️ Mapa do Armazém")
    
    # Criar mapa visual
    fig = go.Figure()
    
    # Coordenadas dinâmicas baseadas no número de classes
    coordenadas = {}
    num_classes = len(sistema.classes)
    for i, classe in enumerate(sistema.classes):
        coordenadas[classe] = (i + 1, 1)
    
    for componente, (x, y) in coordenadas.items():
        quantidade = sistema.inventario[componente]
        loc = sistema.LOCALIZACOES[componente]
        emoji = sistema.EMOJIS[componente]
        
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=50, color='blue' if quantidade > 0 else 'gray'),
            text=[f"{emoji}<br>{componente.upper()}<br>{quantidade} un"],
            textposition="middle center",
            name=f"Setor {loc['setor']} - {loc['zona']}"
        ))
    
    fig.update_layout(
        title="Mapa do Armazém - Sistema Atualizado",
        xaxis=dict(visible=False, range=[0, num_classes + 1]),
        yaxis=dict(visible=False, range=[0, 2]),
        showlegend=True,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detalhes das localizações
    st.subheader("📋 Zonas de Armazenamento")
    for componente, loc in sistema.LOCALIZACOES.items():
        emoji = sistema.EMOJIS[componente]
        with st.expander(f"{emoji} Setor {loc['setor']} - {loc['zona']} ({componente.upper()})"):
            st.write(f"**Corredor:** {loc['corredor']}")
            st.write(f"**Prateleira:** {loc['prateleira']}")
            st.write(f"**Estoque Atual:** {sistema.inventario[componente]} unidades")
            st.write(f"**Instruções:** {loc['instrucoes']}")

# ABA 3: DASHBOARD ATUALIZADO
with tab3:
    st.header("📊 Dashboard Gerencial")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de estoque
        df_estoque = pd.DataFrame([
            {'Componente': f"{sistema.EMOJIS[comp]} {comp.upper()}", 'Quantidade': qtd}
            for comp, qtd in sistema.inventario.items() if qtd > 0
        ])
        
        if not df_estoque.empty:
            fig_estoque = px.bar(
                df_estoque, 
                x='Componente', 
                y='Quantidade',
                title="Estoque por Componente",
                color='Quantidade'
            )
            st.plotly_chart(fig_estoque, use_container_width=True)
        else:
            st.info("📊 Nenhum estoque disponível")
    
    with col2:
        # Gráfico de valor em estoque
        valores_estoque = []
        for componente, quantidade in sistema.inventario.items():
            if quantidade > 0:
                valores_estoque.append({
                    'Componente': f"{sistema.EMOJIS[componente]} {componente.upper()}",
                    'Valor Total': quantidade * sistema.PRECOS[componente]
                })
        
        if valores_estoque:
            df_valores = pd.DataFrame(valores_estoque)
            fig_valores = px.pie(
                df_valores,
                values='Valor Total',
                names='Componente',
                title="Valor Total em Estoque"
            )
            st.plotly_chart(fig_valores, use_container_width=True)
        else:
            st.info("💰 Nenhum valor em estoque")
    
    # Histórico recente
    st.subheader("📈 Histórico de Recebimentos")
    if sistema.historico:
        df_historico = pd.DataFrame(sistema.historico[-10:])
        if not df_historico.empty:
            # Adiciona emoji ao componente
            df_historico['Componente'] = df_historico['componente'].apply(
                lambda x: f"{sistema.EMOJIS.get(x, '🔧')} {x.upper()}"
            )
            
            st.dataframe(
                df_historico[['timestamp', 'Componente', 'numero_serie', 'fornecedor', 'confianca']],
                use_container_width=True
            )
    else:
        st.info("📊 Nenhum recebimento registrado")

# Footer
st.markdown("---")
st.markdown(
    "🏢 **TechLog Solutions** - Sistema Atualizado v3.0 | "
    f"🔧 **Componentes:** {', '.join(sistema.classes)} | "
    "📧 suporte@techlogsolutions.com"
)