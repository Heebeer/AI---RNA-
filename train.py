import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json
from sklearn.utils.class_weight import compute_class_weight

print("🔧 TREINAMENTO ATUALIZADO - USA CLASSES EXISTENTES NO DATASET")

# ================================================================
# CONFIGURAÇÕES
# ================================================================

DATASET_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 50

# ================================================================
# VERIFICAÇÃO DA ESTRUTURA DO DATASET
# ================================================================

print("📊 VERIFICANDO ESTRUTURA DO DATASET...")

if not os.path.exists(DATASET_DIR):
    print(f"❌ Pasta {DATASET_DIR} não existe!")
    exit()

# Lista TODAS as classes que existem no dataset
subdirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
CLASSES_EXISTENTES = sorted(subdirs)

print(f"📁 Subdiretórios encontrados: {CLASSES_EXISTENTES}")

if not CLASSES_EXISTENTES:
    print("❌ Nenhuma classe encontrada no dataset!")
    exit()

# Contar imagens por classe
class_counts = {}
for classe in CLASSES_EXISTENTES:
    class_path = os.path.join(DATASET_DIR, classe)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    class_counts[classe] = len(images)
    print(f"  {classe}: {len(images)} imagens")

min_images = min(class_counts.values())
max_images = max(class_counts.values())
print(f"📈 Mínimo: {min_images}, Máximo: {max_images} imagens por classe")

if min_images < 10:
    print("⚠️  AVISO: Poucas imagens por classe. Recomendado: 15+ imagens")

# ================================================================
# DATA AUGMENTATION OTIMIZADO
# ================================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    channel_shift_range=0.1,
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# ================================================================
# CARREGAMENTO DE DADOS
# ================================================================

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

class_names = list(train_gen.class_indices.keys())
print(f"🎯 Classes para treinamento: {class_names}")

# ================================================================
# ANÁLISE DE BALANCEAMENTO
# ================================================================

print("\n⚖️ ANALISANDO BALANCEAMENTO:")
for i, classe in enumerate(class_names):
    count = np.sum(train_gen.classes == i)
    print(f"  {classe}: {count} imagens (treino)")

total_train = len(train_gen.classes)
print(f"📊 Total de imagens de treino: {total_train}")

# Calcular pesos para balanceamento
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

print("🔢 Pesos calculados para balanceamento:")
for i, classe in enumerate(class_names):
    print(f"  {classe}: {class_weight_dict[i]:.2f}")

# ================================================================
# MODELO OTIMIZADO
# ================================================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Estratégia conservadora - base congelada inicialmente
base_model.trainable = False
print("🧊 FASE 1: Modelo base CONGELADO")

# Arquitetura melhorada
x = GlobalAveragePooling2D()(base_model.output)
x = BatchNormalization()(x)
x = Dense(256, activation="relu", kernel_regularizer=l2(0.001))(x)
x = Dropout(0.4)(x)
x = BatchNormalization()(x)
x = Dense(128, activation="relu", kernel_regularizer=l2(0.001))(x)
x = Dropout(0.3)(x)
output = Dense(len(class_names), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ================================================================
# COMPILAÇÃO
# ================================================================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy", "precision", "recall"]
)

print("📋 Modelo compilado com métricas avançadas")

# ================================================================
# CALLBACKS
# ================================================================

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=12,
    restore_best_weights=True,
    verbose=1,
    min_delta=0.005
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=6,
    min_lr=0.00001,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'model/melhor_modelo_novo.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ================================================================
# TREINAMENTO FASE 1 - BASE CONGELADA
# ================================================================

print("🚀 FASE 1: Treinando com base congelada...")

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=25,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1,
    class_weight=class_weight_dict
)

# ================================================================
# FASE 2 - FINE-TUNING PARCIAL
# ================================================================

print("🔧 FASE 2: Ativando fine-tuning parcial...")

# Descongelar camadas finais gradualmente
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Últimas 30 camadas treináveis
    layer.trainable = False

# Recompilar com LR menor
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy", "precision", "recall"]
)

print(f"🔧 Camadas treináveis: {sum([l.trainable for l in base_model.layers])}/{len(base_model.layers)}")

# Continuar treinamento
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

# Carregar melhor modelo
model.load_weights('model/melhor_modelo_novo.h5')

# Predições
val_gen.reset()
y_pred = model.predict(val_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_gen.classes

# Análise de confiança por classe
print("\n🎯 ANÁLISE DE CONFIANÇA POR CLASSE:")
confusion_analysis = {}

for i, class_name in enumerate(class_names):
    class_mask = y_true == i
    if np.any(class_mask):
        confidences = np.max(y_pred[class_mask], axis=1)
        avg_confidence = float(np.mean(confidences))
        std_confidence = float(np.std(confidences))
        
        # Análise de confusão entre classes
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

# Matriz de confusão detalhada
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred_classes)
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

# Relatório de classificação
print("\n📈 RELATÓRIO DE CLASSIFICAÇÃO:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# ================================================================
# SALVAR MODELO E INFORMAÇÕES (CORRIGIDO)
# ================================================================

model.save("model/modelo_atualizado.h5")
print("✅ Modelo atualizado salvo em model/modelo_atualizado.h5")

# Salvar informações das classes
class_info = {
    'class_names': class_names,
    'class_indices': {k: int(v) for k, v in train_gen.class_indices.items()},
    'class_counts': {k: int(v) for k, v in class_counts.items()},
    'confusion_analysis': confusion_analysis
}

# Função para converter numpy types para Python natives (CORRIGIDA)
def convert_to_serializable(obj):
    if hasattr(obj, 'dtype'):  # É um array numpy
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
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

# Aplicar conversão a todo o class_info
class_info_serializable = convert_to_serializable(class_info)

with open("model/class_info.json", "w") as f:
    json.dump(class_info_serializable, f, indent=2)

with open("model/classes.txt", "w") as f:
    f.write("\n".join(class_names))

print("✅ Informações das classes salvas")

# ================================================================
# GRÁFICOS DE TREINAMENTO
# ================================================================

# Combinar históricos
combined_history = {
    'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
    'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
    'loss': history1.history['loss'] + history2.history['loss'],
    'val_loss': history1.history['val_loss'] + history2.history['val_loss']
}

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(combined_history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(combined_history['val_accuracy'], label='Val Accuracy', linewidth=2)
plt.axvline(x=len(history1.history['accuracy']), color='r', linestyle='--', alpha=0.7, label='Fine-tuning')
plt.title('Evolução da Acurácia')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(combined_history['loss'], label='Train Loss', linewidth=2)
plt.plot(combined_history['val_loss'], label='Val Loss', linewidth=2)
plt.axvline(x=len(history1.history['loss']), color='r', linestyle='--', alpha=0.7, label='Fine-tuning')
plt.title('Evolução da Loss')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
# Gráfico de precisão e recall
if 'precision' in history2.history:
    epochs_range = range(len(history2.history['precision']))
    plt.plot(epochs_range, history2.history['precision'], label='Precision', linewidth=2)
    plt.plot(epochs_range, history2.history['recall'], label='Recall', linewidth=2)
    plt.plot(epochs_range, history2.history['val_precision'], label='Val Precision', linewidth=2, linestyle='--')
    plt.plot(epochs_range, history2.history['val_recall'], label='Val Recall', linewidth=2, linestyle='--')
    plt.title('Precisão e Recall (Fase 2)')
    plt.xlabel('Época')
    plt.ylabel('Métrica')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model/evolucao_treinamento_atualizado.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Gráficos de treinamento salvos")

print("\n🎯 TREINAMENTO CONCLUÍDO!")
print("📋 Classes finais: " + ", ".join(class_names))
print("💡 Modelo pronto para uso no sistema Streamlit")