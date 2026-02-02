# coco_minimal.py
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pycocotools.coco import COCO
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 1) paths 
TRAIN_IMAGES = r"coco2017\train2017"
TRAIN_JSON   = r"coco2017\instances_train2017.json"
VAL_IMAGES   = r"coco2017\val2017"
VAL_JSON     = r"coco2017\instances_val2017.json"

IMG_SIZE = 224
BATCH = 32
EPOCHS = 3

# 2) load COCO -> list of (path, multi-hot label)
def load_items(images_dir, ann_path, max_images=None):
    coco = COCO(ann_path)
    cat_ids = sorted(coco.getCatIds())
    cat_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
    n_classes = len(cat_ids)

    img_ids = coco.getImgIds()
    if max_images:
        img_ids = img_ids[:max_images]

    items = []
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        p = os.path.join(images_dir, info["file_name"])
        if not os.path.exists(p):
            continue

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)

        y = np.zeros((n_classes,), dtype=np.float32)
        for a in anns:
            y[cat_to_idx[a["category_id"]]] = 1.0

        if y.sum() > 0:
            items.append((p, y))

    return items, n_classes

train_items, num_classes = load_items(TRAIN_IMAGES, TRAIN_JSON, max_images=5000)  # możesz zmienić
val_items, _ = load_items(VAL_IMAGES, VAL_JSON, max_images=2000)

# 3) tf.data
def make_ds(items, training):
    paths = [p for p, _ in items]
    labels = np.stack([y for _, y in items], axis=0)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(2000)

    def read_img(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = ds.map(read_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_ds(train_items, training=True)
val_ds   = make_ds(val_items, training=False)

# 4) model (Keras)
base = keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False

model = keras.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="sigmoid")
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.AUC(multi_label=True, name="auc")]
)

# 5) training
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# 6) evaluation + ROC (micro-average)
y_true, y_score = [], []
for x, y in val_ds:
    p = model.predict(x, verbose=0)
    y_true.append(y.numpy())
    y_score.append(p)

y_true = np.concatenate(y_true, axis=0)
y_score = np.concatenate(y_score, axis=0)

fpr, tpr, _ = roc_curve(y_true.ravel(), y_score.ravel())
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve (micro-average) - COCO multi-label")
plt.legend()
plt.grid(True)
plt.show()



#   3/3 ━━━━━━━━━━━━━━━━━━━━ 14s 2s/step - auc: 0.4234 - loss: 0.6672 - val_auc: 0.4251 - val_loss: 0.4740
#   Epoch 2/3
#   3/3 ━━━━━━━━━━━━━━━━━━━━ 3s 1s/step - auc: 0.3965 - loss: 0.4500 - val_auc: 0.4284 - val_loss: 0.3274
#   Epoch 3/3
#   3/3 ━━━━━━━━━━━━━━━━━━━━ 3s 1s/step - auc: 0.4235 - loss: 0.3177 - val_auc: 0.4271 - val_loss: 0.2432