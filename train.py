import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import os

# ================================================================
# CONFIGURAÇÕES
# ================================================================

DATASET_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

os.makedirs("model", exist_ok=True)

# ================================================================
# GERADORES DE TREINO E VALIDAÇÃO
# ================================================================

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# Salva a ordem das classes
class_names = list(train_gen.class_indices.keys())
with open("model/classes.txt", "w") as f:
    f.write("\n".join(class_names))

print("Classes detectadas:", class_names)

# ================================================================
# MODELO BASE (MobileNetV2)
# ================================================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # congela pesos

# ================================================================
# CABEÇALHO DA REDE NEURAL
# ================================================================

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
output = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================================================================
# TREINAMENTO
# ================================================================

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ================================================================
# SALVA O MODELO TREINADO
# ================================================================

model.save("model/modelo_componentes.h5")
print("Modelo salvo em model/modelo_componentes.h5")

# ================================================================
# GRÁFICO DE TREINAMENTO
# ================================================================

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Treino")
plt.plot(history.history["val_accuracy"], label="Validação")
plt.title("Acurácia")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Treino")
plt.plot(history.history["val_loss"], label="Validação")
plt.title("Perda")
plt.legend()

plt.savefig("model/treinamento.png")
print("Gráfico salvo em model/treinamento.png")

# ================================================================
# MATRIZ DE CONFUSÃO
# ================================================================

print("\nGerando matriz de confusão...")

val_gen.reset()
pred = model.predict(val_gen)
y_pred = np.argmax(pred, axis=1)
y_true = val_gen.classes

cm = confusion_matrix(y_true, y_pred)

print("\nMatriz de confusão:\n", cm)
print("\nRelatório de classificação:")
print(classification_report(y_true, y_pred, target_names=class_names))