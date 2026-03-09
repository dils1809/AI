import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow import keras
from tensorflow.keras import layers


# Cargar datos
digits = load_digits()
X = digits.data
y = digits.target

# Normalización
X = X / 16.0

# División train-test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Modelo
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(64,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilación
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluación
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Predicciones
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()

# Métricas
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Ejemplos correctos
correct = np.where(y_test == y_pred_classes)[0]

plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_test[correct[i]].reshape(8, 8), cmap='gray')
    plt.title(f"{y_pred_classes[correct[i]]}")
    plt.axis('off')
plt.show()

# Ejemplos incorrectos
incorrect = np.where(y_test != y_pred_classes)[0]

plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_test[incorrect[i]].reshape(8, 8), cmap='gray')
    plt.title(f"P:{y_pred_classes[incorrect[i]]} R:{y_test[incorrect[i]]}")
    plt.axis('off')
plt.show()