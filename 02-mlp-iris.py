# 1) Importy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# 2) Wczytanie danych Iris
data = load_iris()
X = data.data          # 4 cechy: sepal length/width, petal length/width
y = data.target        # klasy: 0,1,2 (setosa, versicolor, virginica)
class_names = data.target_names  # nazwy klas


# 3) Podział na train/test (stratify utrzymuje proporcje klas)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4) Standaryzacja cech (ważne dla MLP)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5) One-hot encoding etykiet (bo 3 klasy + softmax)
num_classes = 3
y_train_oh = keras.utils.to_categorical(y_train, num_classes)
y_test_oh = keras.utils.to_categorical(y_test, num_classes)


# 6) Bardziej złożony model (MLP) dla multi-class
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),     # 4 cechy
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.25),

    layers.Dense(32, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.20),

    layers.Dense(16, activation="relu"),
    layers.Dense(num_classes, activation="softmax")  # 3 klasy
])

# 7) Kompilacja
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 8) Trening
history = model.fit(
    X_train, y_train_oh,
    validation_split=0.2,
    epochs=80,
    batch_size=16,
    verbose=1
)

# 9) Ocena na zbiorze testowym
test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
print("\n=== Wyniki na zbiorze testowym ===")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss    : {test_loss:.4f}")

# 10) Predykcje + confusion matrix
y_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:")
print(cm)

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# 11) Wizualizacja: confusion matrix
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix (Iris)")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.xticks([0, 1, 2], class_names, rotation=20)
plt.yticks([0, 1, 2], class_names)

for (i, j), val in np.ndenumerate(cm):
    plt.text(j, i, str(val), ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.show()

# 12) Wykresy treningu
plt.figure()
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.title("Accuracy during training")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Loss during training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()
