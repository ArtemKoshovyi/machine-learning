import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

DATA_DIR = "raw_dataset"   
IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42

# 1) pobranie train/val 
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.30,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH
)

val_test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.30,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH
)

class_names = train_ds.class_names
num_classes = len(class_names)

# 2) żeby szybciej
val_batches = tf.data.experimental.cardinality(val_test_ds).numpy()
test_ds = val_test_ds.take(val_batches // 2)
val_ds = val_test_ds.skip(val_batches // 2)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# 3) Model VGG16 + transfer learning
base = tf.keras.applications.VGG16(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
base.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.vgg16.preprocess_input(inputs)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=5)

# 4) Fine-tuning 
base.trainable = True
for layer in base.layers[:-6]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(train_ds, validation_data=val_ds, epochs=3)

# 5) Ocena
loss, acc = model.evaluate(test_ds)
print("Test accuracy:", acc)

# 6) Confusion matrix
y_true = []
y_pred = []

for images, labels in test_ds:
    probs = model.predict(images, verbose=0)
    preds = np.argmax(probs, axis=1)
    y_true.extend(labels.numpy().tolist())
    y_pred.extend(preds.tolist())

cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
plt.figure(figsize=(10, 10))
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(include_values=False, xticks_rotation=90)
plt.tight_layout()
plt.show()




#  11/11 ━━━━━━━━━━━━━━━━━━━━ 57s 5s/step - accuracy: 0.1225 - loss: 7.0343 - val_accuracy: 0.2289 - val_loss: 3.8871
#  Epoch 2/5
#  11/11 ━━━━━━━━━━━━━━━━━━━━ 58s 5s/step - accuracy: 0.2133 - loss: 5.5663 - val_accuracy: 0.3373 - val_loss: 3.0510
#  Epoch 3/5
#  11/11 ━━━━━━━━━━━━━━━━━━━━ 56s 5s/step - accuracy: 0.2408 - loss: 4.5896 - val_accuracy: 0.3614 - val_loss: 2.5570
#  Epoch 4/5
#  11/11 ━━━━━━━━━━━━━━━━━━━━ 59s 5s/step - accuracy: 0.3198 - loss: 3.5606 - val_accuracy: 0.4819 - val_loss: 2.1642
#  Epoch 5/5
#  11/11 ━━━━━━━━━━━━━━━━━━━━ 62s 6s/step - accuracy: 0.3902 - loss: 2.9779 - val_accuracy: 0.4217 - val_loss: 2.3497
#  Epoch 1/3
#  11/11 ━━━━━━━━━━━━━━━━━━━━ 73s 7s/step - accuracy: 0.4140 - loss: 2.4698 - val_accuracy: 0.5663 - val_loss: 1.4484
#  Epoch 2/3
#  11/11 ━━━━━━━━━━━━━━━━━━━━ 76s 7s/step - accuracy: 0.5347 - loss: 1.5119 - val_accuracy: 0.6627 - val_loss: 1.2636
#  Epoch 3/3
#  11/11 ━━━━━━━━━━━━━━━━━━━━ 81s 7s/step - accuracy: 0.5977 - loss: 1.1868 - val_accuracy: 0.6145 - val_loss: 1.4356
#  2/2 ━━━━━━━━━━━━━━━━━━━━ 8s 4s/step - accuracy: 0.6458 - loss: 1.2656


#  Test accuracy: 0.640625
#  2026-02-01 17:37:11.250297: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: 
#  OUT_OF_RANGE: End of sequence
# 