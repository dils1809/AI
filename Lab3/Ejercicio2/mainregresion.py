import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow import keras
from tensorflow.keras import layers


# Crear estructura carpetas
base_path = "resultados_regresion"
img_path = os.path.join(base_path, "imagenes")
metrics_path = os.path.join(base_path, "metricas")

os.makedirs(img_path, exist_ok=True)
os.makedirs(metrics_path, exist_ok=True)


# Cargar dataset
housing = fetch_california_housing()

X = housing.data
y = housing.target


# División train-test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Escalamiento obligatorio
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Modelo más profundo
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # salida regresión
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)


# Entrenamiento
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# Evaluación
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)


# Guardar métricas
with open(os.path.join(metrics_path, "metricas_regresion.txt"), "w") as f:
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")


print("\nMSE:", mse)
print("MAE:", mae)
print("RMSE:", rmse)


# Scatter Plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.title("Predicciones vs Valores Reales")
plt.tight_layout()
plt.savefig(os.path.join(img_path, "scatter_plot.png"))
plt.close()


# 3 Nuevas Predicciones
nuevos_datos = np.array([
    X_test[0],
    X_test[1],
    X_test[2]
])

predicciones_nuevas = model.predict(nuevos_datos)

with open(os.path.join(metrics_path, "predicciones_nuevas.txt"), "w") as f:
    for i, pred in enumerate(predicciones_nuevas):
        f.write(f"Observación {i+1}: {pred[0]:.4f}\n")

print("\n3 nuevas predicciones generadas.")

print("\nResultados guardados en carpeta 'resultados_regresion'")

# 3 Observaciones completamente nuevas

nuevas_observaciones = np.array([
    [8.0, 30.0, 6.0, 1.0, 1000.0, 3.0, 34.0, -118.0],
    [4.5, 20.0, 5.0, 1.2, 800.0, 2.5, 36.0, -120.0],
    [10.0, 10.0, 7.0, 1.0, 1500.0, 4.0, 37.0, -122.0]
])

# Escalar con el mismo scaler entrenado
nuevas_observaciones = scaler.transform(nuevas_observaciones)

pred_nuevas = model.predict(nuevas_observaciones)

print("\nPredicciones para nuevas observaciones:")
for i, pred in enumerate(pred_nuevas):
    print(f"Observación nueva {i+1}: {pred[0]:.4f}")