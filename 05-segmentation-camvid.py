# camvid_fast_segmentation.py
import os, glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ---------------------------
# 0) Settings + paths
# ---------------------------
ROOT = "CamVid"
IMG_H, IMG_W = 192, 192
BATCH = 16
EPOCHS = 10

TRAIN_I, TRAIN_M = f"{ROOT}/train", f"{ROOT}/trainannot"
VAL_I,   VAL_M   = f"{ROOT}/val",   f"{ROOT}/valannot"
TEST_I,  TEST_M  = f"{ROOT}/test",  f"{ROOT}/testannot"

# ---------------------------
# 1) List files
# ---------------------------
def list_imgs(folder):
    files = []
    for e in ("*.png", "*.jpg", "*.jpeg"):
        files += glob.glob(os.path.join(folder, e))
    return sorted(files)

tr_i, tr_m = list_imgs(TRAIN_I), list_imgs(TRAIN_M)
va_i, va_m = list_imgs(VAL_I),   list_imgs(VAL_M)
te_i, te_m = list_imgs(TEST_I),  list_imgs(TEST_M)

assert len(tr_i) == len(tr_m), "train counts differ"
assert len(va_i) == len(va_m), "val counts differ"
assert len(te_i) == len(te_m), "test counts differ"

# ---------------------------
# 2) Detect mask type + NUM_CLASSES
# ---------------------------
def read_mask_np(path):
    raw = tf.io.read_file(path)
    return tf.image.decode_png(raw, channels=0).numpy()

sample_mask = read_mask_np(tr_m[0])
is_rgb_mask = (sample_mask.ndim == 3 and sample_mask.shape[-1] == 3)

if is_rgb_mask:
    # Build palette from train masks
    import numpy as np
    colors_set = set()
    for p in tr_m:
        m = read_mask_np(p)
        uniq = np.unique(m.reshape(-1, 3), axis=0)
        for c in uniq:
            colors_set.add((int(c[0]), int(c[1]), int(c[2])))

    colors = np.array(sorted(list(colors_set)), dtype=np.uint8)
    NUM_CLASSES = colors.shape[0]
    COLORS_TF = tf.constant(colors, dtype=tf.uint8)

    def rgb_to_class(mask_rgb_uint8):
        m = tf.expand_dims(mask_rgb_uint8, axis=-2)       # (H,W,1,3)
        c = tf.reshape(COLORS_TF, (1, 1, NUM_CLASSES, 3)) # (1,1,C,3)
        eq = tf.reduce_all(tf.equal(m, c), axis=-1)        # (H,W,C)
        cls = tf.argmax(tf.cast(eq, tf.int32), axis=-1)    # (H,W)
        return cls
else:
    max_val = 0
    for p in tr_m:
        m = read_mask_np(p)
        if m.ndim == 3:
            m = m[..., 0]
        max_val = max(max_val, int(m.max()))
    NUM_CLASSES = max_val + 1

print("Mask type:", "RGB(color)" if is_rgb_mask else "Indexed(grayscale)")
print("NUM_CLASSES =", NUM_CLASSES)

# ---------------------------
# 3) tf.data pipeline
# ---------------------------
def load_pair(ip, mp):
    img = tf.image.decode_png(tf.io.read_file(ip), channels=3)
    img = tf.image.resize(img, (IMG_H, IMG_W))
    img = tf.cast(img, tf.float32) / 255.0

    raw = tf.io.read_file(mp)
    if is_rgb_mask:
        mask = tf.image.decode_png(raw, channels=3)
        mask = tf.image.resize(mask, (IMG_H, IMG_W), method="nearest")
        mask = tf.cast(mask, tf.uint8)
        mask = rgb_to_class(mask)
    else:
        mask = tf.image.decode_png(raw, channels=1)
        mask = tf.image.resize(mask, (IMG_H, IMG_W), method="nearest")
        mask = tf.cast(mask, tf.int32)
        mask = tf.squeeze(mask, -1)
    return img, mask

def make_ds(imgs, masks, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((imgs, masks))
    ds = ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(512)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_ds(tr_i, tr_m, shuffle=True)
val_ds   = make_ds(va_i, va_m)
test_ds  = make_ds(te_i, te_m)

# ---------------------------
# 4) Model: MobileNetV2 encoder + light decoder
# ---------------------------
def fast_seg_model(num_classes):
    inp = keras.Input((IMG_H, IMG_W, 3))
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=inp
    )
    base.trainable = False  # fast training

    x = base.output  # ~ (H/32, W/32)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)

    # upsample back to input resolution
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)

    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)

    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)

    x = layers.UpSampling2D(2)(x)
    x = layers.UpSampling2D(2)(x)  # total x32

    out = layers.Conv2D(num_classes, 1, activation="softmax")(x)
    return keras.Model(inp, out)

class MeanIoUFromProbs(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

model = fast_seg_model(NUM_CLASSES)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[MeanIoUFromProbs(num_classes=NUM_CLASSES)]
)

# ---------------------------
# 5) Train
# ---------------------------
cb = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best_fast_camvid.keras", save_best_only=True, monitor="val_loss"),
]

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb)

# ---------------------------
# 6) Evaluate
# ---------------------------
print("\nTest evaluation:")
test_res = model.evaluate(test_ds, verbose=1)
print("Loss =", test_res[0], "| MeanIoU =", test_res[1])

# ---------------------------
# 7) Visualize predictions (few test images)
# ---------------------------
def show_examples(ds, n=3):
    taken = 0
    for imgs, masks in ds:
        preds = model.predict(imgs, verbose=0)
        preds_cls = tf.argmax(preds, axis=-1)

        for i in range(imgs.shape[0]):
            if taken >= n:
                return

            img = imgs[i].numpy()
            gt = masks[i].numpy()
            pr = preds_cls[i].numpy()

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1); plt.title("Image"); plt.imshow(img); plt.axis("off")
            plt.subplot(1, 3, 2); plt.title("GT mask"); plt.imshow(gt); plt.axis("off")
            plt.subplot(1, 3, 3); plt.title("Pred mask"); plt.imshow(pr); plt.axis("off")
            plt.show()

            taken += 1

show_examples(test_ds, n=3)



#  implementation if present.
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 9s 2s/step - loss: 2.7679 - mean_io_u_from_probs: 0.0771 - val_loss: 2.2651 - val_mean_io_u_from_probs: 0.0917
# Epoch 2/10
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 2s 889ms/step - loss: 2.2930 - mean_io_u_from_probs: 0.1135 - val_loss: 1.7778 - val_mean_io_u_from_probs: 0.1204
# Epoch 3/10
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 2s 880ms/step - loss: 1.3690 - mean_io_u_from_probs: 0.1924 - val_loss: 1.4540 - val_mean_io_u_from_probs: 0.1767
# Epoch 4/10
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 2s 639ms/step - loss: 1.1418 - mean_io_u_from_probs: 0.2514 - val_loss: 1.4965 - val_mean_io_u_from_probs: 0.1482
# Epoch 5/10
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 2s 877ms/step - loss: 1.0371 - mean_io_u_from_probs: 0.2842 - val_loss: 1.2803 - val_mean_io_u_from_probs: 0.1998
# Epoch 6/10
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 2s 627ms/step - loss: 0.9220 - mean_io_u_from_probs: 0.2986 - val_loss: 1.3044 - val_mean_io_u_from_probs: 0.1881
# Epoch 7/10
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 2s 885ms/step - loss: 0.8845 - mean_io_u_from_probs: 0.3112 - val_loss: 1.1394 - val_mean_io_u_from_probs: 0.2276
# Epoch 8/10
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 2s 900ms/step - loss: 0.8321 - mean_io_u_from_probs: 0.3237 - val_loss: 1.0642 - val_mean_io_u_from_probs: 0.2511
# Epoch 9/10
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 2s 623ms/step - loss: 0.7791 - mean_io_u_from_probs: 0.3354 - val_loss: 1.0742 - val_mean_io_u_from_probs: 0.2433
# Epoch 10/10
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 2s 899ms/step - loss: 0.7677 - mean_io_u_from_probs: 0.3413 - val_loss: 1.0284 - val_mean_io_u_from_probs: 0.2521

# Test evaluation:
#  3/3 ━━━━━━━━━━━━━━━━━━━━ 1s 206ms/step - loss: 1.0827 - mean_io_u_from_probs: 0.2925
#  Loss = 1.1104700565338135 | MeanIoU = 0.2802012860774994