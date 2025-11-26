import tensorflow as tf  # Importa TensorFlow (contém Keras integrado)
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Gerador com augmentation
from tensorflow.keras.applications import MobileNetV2  # Modelo pré-treinado (ImageNet)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model  # Para construir o modelo funcional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam  # Otimizador Adam
from tensorflow.keras.regularizers import l2  # Regularização L2 para camadas Dense
import numpy as np  # Biblioteca numérica
import os  # Operações de arquivo/sistema
from PIL import Image  # Manipulação de imagens (se necessário)
import matplotlib.pyplot as plt  # Plotagem de gráficos
from sklearn.metrics import confusion_matrix, classification_report  # Métricas de avaliação
import seaborn as sns  # Visualização (heatmap)
import json  # Salvar informações em JSON
from sklearn.utils.class_weight import compute_class_weight  # Para calcular pesos de classe

print("TREINAMENTO INICIALIZADO")  # Log inicial

# ================================================================
# CONFIGURAÇÕES
# ================================================================

DATASET_DIR = "dataset"  # Pasta raiz do dataset (subpastas = classes)
IMG_SIZE = (224, 224)  # Tamanho fixo das imagens (compatível com MobileNetV2)
BATCH_SIZE = 8  # Tamanho do batch; pequeno para economizar memória
EPOCHS = 50  # Número máximo de épocas (não usado diretamente no fit dividido)

# ================================================================
# VERIFICAÇÃO DA ESTRUTURA DO DATASET
# ================================================================

# Mensagem informativa sobre verificação do dataset
print("VERIFICANDO ESTRUTURA DO DATASET...")

# Verifica se o diretório existe; se não existir, encerra o programa
if not os.path.exists(DATASET_DIR):
    print(f"❌ Pasta {DATASET_DIR} não existe!")
    exit()

# Lista subpastas dentro do dataset — cada subpasta é tratada como uma classe
subdirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
CLASSES_EXISTENTES = sorted(subdirs)  # Ordena alfabeticamente as classes

# os.listdir retorna nomes (strings) dos arquivos/pastas presentes em DATASET_DIR

print(f"Subdiretórios encontrados: {CLASSES_EXISTENTES}")  # Exibe as classes encontradas

# Se não houver classes, encerra o script
if not CLASSES_EXISTENTES:
    print("❌ Nenhuma classe encontrada no dataset!")
    exit()

# Conta quantas imagens (jpg/jpeg/png) existem em cada pasta de classe
class_counts = {}
for classe in CLASSES_EXISTENTES:
    class_path = os.path.join(DATASET_DIR, classe)  # Caminho completo para a pasta da classe
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    class_counts[classe] = len(images)  # Armazena a contagem
    print(f"  {classe}: {len(images)} imagens")  # Exibe contagem por classe

# Calcula o mínimo e máximo de imagens entre classes para avaliar balanceamento
min_images = min(class_counts.values())
max_images = max(class_counts.values())
print(f"📈 Mínimo: {min_images}, Máximo: {max_images} imagens por classe")

# Alerta se alguma classe tiver poucas imagens
if min_images < 10:
    print("⚠️  AVISO: Poucas imagens por classe. Recomendado: 15+ imagens")

# ================================================================
# DATA AUGMENTATION OTIMIZADO - cria gerador de imagem para o treino e validação
# ================================================================

# Cria um ImageDataGenerator para treino com várias transformações de data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normaliza pixels para a faixa [0, 1]
    rotation_range=25,  # Rotaciona imagens aleatoriamente até ±25 graus
    width_shift_range=0.2,  # Translação horizontal aleatória
    height_shift_range=0.2,  # Translação vertical aleatória
    shear_range=0.2,  # Transformação de shear
    zoom_range=0.25,  # Zoom aleatório
    horizontal_flip=True,  # Flip horizontal (útil para muitas tarefas de visão)
    vertical_flip=True,  # Flip vertical (usar com cautela dependendo do problema)
    brightness_range=[0.8, 1.2],  # Ajusta brilho aleatoriamente
    channel_shift_range=0.1,  # Deslocamento nos canais de cor
    fill_mode='nearest',  # Modo de preenchimento de pixels fora dos limites
    validation_split=0.2  # Reserva 20% dos dados para validação via subset
)

# Gerador de validação para validação (não embaralha e não aplica augmentations pesadas)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# ================================================================
# CARREGAMENTO DE DADOS
# ================================================================

# Gerador de validação para treinamento
train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,  # pasta raiz
    target_size=IMG_SIZE,  # redimensiona imagem para IMG_SIZE
    batch_size=BATCH_SIZE,  # tamanho do lote
    class_mode="categorical",  # rótulos como one-hot (multi-classe)
    subset="training",  # pega as amostras de treino (80%)
    shuffle=True  # embaralha os dados a cada época
)

# Gerador de validação (não embaralha para manter ordenação durante avaliação)
val_gen = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",  # pega as amostras de validação (20%)
    shuffle=False
)

# Lista com os nomes das classes (ordem definida por class_indices)
class_names = list(train_gen.class_indices.keys())
print(f"🎯 Classes para treinamento: {class_names}")

# ================================================================
# ANÁLISE DE BALANCEAMENTO
# ================================================================

print("\n⚖️ ANALISANDO BALANCEAMENTO:")
for i, classe in enumerate(class_names):
    # train_gen.classes contém os índices das classes para todas as amostras de treino
    count = np.sum(train_gen.classes == i)
    print(f"  {classe}: {count} imagens (treino)")

# Total de imagens no conjunto de treino
total_train = len(train_gen.classes)
print(f"📊 Total de imagens de treino: {total_train}")

# Calcula pesos para cada classe para compensar desbalanceamento
class_weights = compute_class_weight(
    'balanced',  # estratégia balanced que inversamente pondera pela frequência
    classes=np.unique(train_gen.classes),  # classes únicas (índices)
    y=train_gen.classes  # vetor de labels do gerador
)
# Converte para dicionário esperado pelo model.fit (índice -> peso)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

print("🔢 Pesos calculados para balanceamento:")
for i, classe in enumerate(class_names):
    print(f"  {classe}: {class_weight_dict[i]:.2f}")

# ================================================================
# MODELO OTIMIZADO
# ================================================================

# Carrega MobileNetV2 sem a parte superior (include_top=False) e com pesos do ImageNet
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
# >> Estamos reaproveitando uma rede já treinada e congelando suas camadas para treinar apenas a parte que criamos, evitando destruir o conhecimento prévio.

# Estratégia conservadora - congela toda a base inicialmente
base_model.trainable = False
print("🧊 FASE 1: Modelo base CONGELADO")

# Constrói a nova cabeça (head) do modelo usando API funcional
x = GlobalAveragePooling2D()(base_model.output)  # Reduce spatial dims para um vetor
x = BatchNormalization()(x)  # Normaliza ativação para acelerar convergência
x = Dense(256, activation="relu", kernel_regularizer=l2(0.001))(x)  # Camada densa com L2
x = Dropout(0.4)(x)  # Dropout para reduzir overfitting
x = BatchNormalization()(x)  # Outra normalização
x = Dense(128, activation="relu", kernel_regularizer=l2(0.001))(x)  # Camada intermediária
x = Dropout(0.3)(x)  # Dropout adicional
output = Dense(len(class_names), activation="softmax")(x)  # Saída multi-classe (softmax)

# Cria o modelo final ligando a base pré-treinada e a cabeça customizada
model = Model(inputs=base_model.input, outputs=output)

# >> Criamos um novo topo (head) personalizado para adaptar a MobileNet às classes do nosso dataset.


# ================================================================
# COMPILAÇÃO
# ================================================================

# Compila o modelo com Adam e loss apropriada para múltiplas classes
model.compile(
    optimizer=Adam(learning_rate=0.001),  # LR inicial relativamente padrão
    loss="categorical_crossentropy",
    metrics=["accuracy"]  # Métrica principal para monitoramento
)

# >> Aqui definimos as regras de aprendizado do modelo.


print("📋 Modelo compilado com métricas básicas")

# ================================================================
# CALLBACKS
# ================================================================

# EarlyStopping para interromper se a métrica de validação não melhorar
early_stop = EarlyStopping(
    monitor='val_accuracy',  # monitora acurácia de validação
    patience=12,  # espera 12 épocas sem melhora antes de parar
    restore_best_weights=True,  # retorna os melhores pesos no final
    verbose=1,
    min_delta=0.005  # só considera melhora se maior que 0.5%
)

# ReduceLROnPlateau reduz o Learning Rate quando a val_loss estaciona
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # multiplica LR por 0.5
    patience=6,  # espera 6 épocas
    min_lr=0.00001,  # limite inferior do LR
    verbose=1
)

# ModelCheckpoint salva o melhor modelo baseado em val_accuracy
checkpoint = ModelCheckpoint(
    'model/melhor_modelo_novo.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ================================================================
# TREINAMENTO FASE 1 - BASE CONGELADA
# ================================================================

#Nesta etapa, o modelo MobileNetV2 pré-treinado está com todas as suas camadas congeladas, ou seja:
#Ele não atualiza os pesos da rede base.
#Apenas a parte nova que você adicionou (a “cabeça” do modelo) será treinada.
#Isso permite que o modelo aprenda suas classes (seu dataset) sem estragar o conhecimento já adquirido no ImageNet.

print("🚀 FASE 1: Treinando com base congelada...")

history1 = model.fit( # Esta chamada inicia o treinamento real do modelo.
    train_gen, #E o gerador que fornece as imagens do treino, já normalizadas e aumentadas (augmented).
    validation_data=val_gen, #Conjunto de validação — o modelo usa para medir desempenho entre épocas.
    epochs=25,  # Tentará até 25, mas pode parar antes por EarlyStopping
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1,
    class_weight=class_weight_dict  # Aplica pesos de classe para balanceamento
)


# ================================================================
# FASE 2 - FINE-TUNING PARCIAL
# ================================================================

print("🔧 FASE 2: Ativando fine-tuning parcial...")

# Descongela a base parcialmente: marca todas como treináveis e re-congela as primeiras camadas
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Mantém congeladas todas exceto as últimas 30
    layer.trainable = False

# Recompila com LR menor para fine-tuning suave
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Exibe quantas camadas da base estão treináveis
print(f"🔧 Camadas treináveis: {sum([l.trainable for l in base_model.layers])}/{len(base_model.layers)}")

# Continua o treinamento (ajuste fino) com mesmas callbacks
history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=25,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1,
    class_weight=class_weight_dict
)

# ================================================================
# AVALIAÇÃO DETALHADA
# ================================================================

print("\n📊 AVALIAÇÃO FINAL DO MODELO")

# Carrega os melhores pesos salvos pelo checkpoint
model.load_weights('model/melhor_modelo_novo.h5')

# Gera previsões sobre o conjunto de validação
val_gen.reset()  # garante que o gerador comece do início
y_pred = model.predict(val_gen)  # probabilidades por classe
y_pred_classes = np.argmax(y_pred, axis=1)  # índice da classe prevista
y_true = val_gen.classes  # rótulos verdadeiros

# Análise de confiança por classe — média e desvio padrão das probabilidades
print("\n🎯 ANÁLISE DE CONFIANÇA POR CLASSE:")
confusion_analysis = {}

for i, class_name in enumerate(class_names):
    class_mask = y_true == i  # seleciona amostras da classe i
    if np.any(class_mask):
        confidences = np.max(y_pred[class_mask], axis=1)  # confiança das predições para essa classe
        avg_confidence = float(np.mean(confidences))
        std_confidence = float(np.std(confidences))

        # Analisa com quais outras classes essa classe está sendo confundida
        pred_for_this_class = y_pred_classes[class_mask]
        confusion_with = {}
        for j, other_class in enumerate(class_names):
            if i != j:
                count = int(np.sum(pred_for_this_class == j))
                if count > 0:
                    confusion_with[other_class] = count

        confusion_analysis[class_name] = {
            'avg_confidence': avg_confidence,
            'std_confidence': std_confidence,
            'confusion_with': confusion_with
        }

        print(f"  {class_name}:")
        print(f"    Confiança média: {avg_confidence:.2%} ± {std_confidence:.2%}")
        if confusion_with:
            print(f"    Confusões: {confusion_with}")

# ================================================================
# MATRIZ DE CONFUSÃO
# ================================================================

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred_classes)  # matriz de confusão (contagens)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Quantidade'})
plt.title('Matriz de Confusão - Classes Existentes')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig('model/matriz_confusao_atualizada.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Matriz de confusão salva")

# Relatório de classificação com precision, recall e f1-score
print("\n📈 RELATÓRIO DE CLASSIFICAÇÃO:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# ================================================================
# SALVAR MODELO E INFORMAÇÕES
# ================================================================

# Salva o modelo final (arquitetura + pesos)
model.save("model/modelo_atualizado.h5")
print("✅ Modelo atualizado salvo em model/modelo_atualizado.h5")

# Prepara informações adicionais sobre classes e contagens para salvar em JSON
class_info = {
    'class_names': class_names,
    'class_indices': {k: int(v) for k, v in train_gen.class_indices.items()},
    'class_counts': {k: int(v) for k, v in class_counts.items()},
    'confusion_analysis': confusion_analysis
}

# Função recursiva para converter objetos numpy (e outros) em tipos nativos Python para JSON
def convert_to_serializable(obj):
    if hasattr(obj, 'dtype'):
        # Casos de arrays numpy
        if np.issubdtype(obj.dtype, np.integer):
            return int(obj)
        elif np.issubdtype(obj.dtype, np.floating):
            return float(obj)
        elif np.issubdtype(obj.dtype, np.bool_):
            return bool(obj)
        else:
            return obj.tolist()
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        return float(obj)
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

# Aplica a conversão para garantir serialização correta
class_info_serializable = convert_to_serializable(class_info)

# Grava o JSON com informações das classes e análises
with open("model/class_info.json", "w") as f:
    json.dump(class_info_serializable, f, indent=2)

# Também grava uma lista simples de classes em texto
with open("model/classes.txt", "w") as f:
    f.write("\n".join(class_names))

print("✅ Informações das classes salvas")

# ================================================================
# GRÁFICOS DE TREINAMENTO
# ================================================================

# Combina históricos das duas fases (fase 1 + fase 2) para plotagem contínua
combined_history = {
    'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
    'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
    'loss': history1.history['loss'] + history2.history['loss'],
    'val_loss': history1.history['val_loss'] + history2.history['val_loss']
}

plt.figure(figsize=(15, 5))

# Subplot 1: Acurácia
plt.subplot(1, 3, 1)
plt.plot(combined_history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(combined_history['val_accuracy'], label='Val Accuracy', linewidth=2)
# Linha vertical que marca o início do fine-tuning (após history1)
plt.axvline(x=len(history1.history['accuracy']), color='r', linestyle='--', alpha=0.7, label='Fine-tuning')
plt.title('Evolução da Acurácia')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Loss
plt.subplot(1, 3, 2)
plt.plot(combined_history['loss'], label='Train Loss', linewidth=2)
plt.plot(combined_history['val_loss'], label='Val Loss', linewidth=2)
plt.axvline(x=len(history1.history['loss']), color='r', linestyle='--', alpha=0.7, label='Fine-tuning')
plt.title('Evolução da Loss')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 3: Texto informativo sobre métricas adicionais
plt.subplot(1, 3, 3)
plt.text(0.5, 0.5, 'Métricas de Precision/Recall\ncalculadas no relatório final\n(classification_report)', 
         ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
plt.title('Métricas Adicionais')
plt.axis('off')

plt.tight_layout()
plt.savefig('model/evolucao_treinamento_atualizado.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Gráficos de treinamento salvos")

print("\n🎯 TREINAMENTO CONCLUÍDO!")
print("📋 Classes finais: " + ", ".join(class_names))
print("💡 Modelo pronto para uso no sistema Streamlit")
