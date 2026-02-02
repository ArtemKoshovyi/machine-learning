# 1) Importy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# 2) Wczytanie danych (Breast Cancer Dataset z sklearn)
data = load_breast_cancer()
X = data.data                 # cechy (30 kolumn)
y = data.target               # etykieta: 0 = malignant, 1 = benign (w tym zbiorze)

# Dla czytelności: uznajmy, że "positive" = malignant (złośliwy)
# W danych: malignant = 0, benign = 1, więc odwracamy etykiety:
y_pos = (y == 0).astype(int)  # 1 = malignant (pozytywna), 0 = benign (negatywna)


# 3) Podział na train/test (z zachowaniem proporcji klas)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_pos,
    test_size=0.2,
    random_state=42,
    stratify=y_pos
)

# 4) Standaryzacja (ważne dla sieci MLP)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 5) Budowa prostego modelu (MLP)
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),  # 30 cech
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")     # wyjście binarne (0..1)
])

# 6) Kompilacja
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 7) Trening
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

# 8) Predykcja na teście (prawdopodobieństwa -> klasy)
y_prob = model.predict(X_test).ravel()
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

# 9) Metryki: accuracy, precision, recall(czułość)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)  # czułość (TPR)

# 10) Confusion matrix i specificity (swoistość = TN / (TN + FP))
cm = confusion_matrix(y_test, y_pred)  # układ: [[TN, FP], [FN, TP]]
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

print("=== Wyniki na zbiorze testowym ===")
print(f"Accuracy     : {acc:.4f}")
print(f"Precision    : {prec:.4f}")
print(f"Recall/TPR   : {recall:.4f}  (czułość)")
print(f"Specificity  : {specificity:.4f}  (swoistość)")
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(cm)

# 11) Wizualizacja: macierz pomyłek (heatmap)
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix (threshold=0.5)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks([0, 1], ["Negative (benign)", "Positive (malignant)"])
plt.yticks([0, 1], ["Negative (benign)", "Positive (malignant)"])

# wartości w komórkach
for (i, j), val in np.ndenumerate(cm):
    plt.text(j, i, str(val), ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.show()

# 12) (Opcjonalnie) wykres treningu: loss / accuracy
plt.figure()
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Loss during training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.title("Accuracy during training")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()
