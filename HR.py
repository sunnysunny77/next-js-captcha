import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

tf.keras.mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(True)

BATCH_SIZE = 512
NUM_CLASSES = 26

df_train = pd.read_csv("./emnist-byclass-train.csv", header=None)
df_test = pd.read_csv("./emnist-byclass-test.csv", header=None)

X_train = df_train.drop(columns=[0]).to_numpy()
y_train = df_train[0].to_numpy()
X_test = df_test.drop(columns=[0]).to_numpy()
y_test = df_test[0].to_numpy()

train_mask = (y_train >= 10) & (y_train <= 35)
test_mask = (y_test >= 10) & (y_test <= 35)

X_train = X_train[train_mask]
y_train = y_train[train_mask]
X_test = X_test[test_mask]
y_test = y_test[test_mask]

y_train -= 10
y_test -= 10

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

def fix_orientation(images):
    return np.flip(np.rot90(images, k=3, axes=(1, 2)), axis=2)

X_train = fix_orientation(X_train)
X_test = fix_orientation(X_test)

X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

y_train = to_categorical(y_train, num_classes=NUM_CLASSES).astype(np.float32)
y_val = to_categorical(y_val, num_classes=NUM_CLASSES).astype(np.float32)
y_test = to_categorical(y_test, num_classes=NUM_CLASSES).astype(np.float32)

augmentation = Sequential([
    layers.RandomRotation(0.05),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomZoom(0.05),
    layers.RandomContrast(0.05)
])

def prepare_augmentation(x, y):
    x = augmentation(x, training=True)
    return x, y

train = (
    Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(65536)
    .batch(BATCH_SIZE)
    .map(prepare_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

val = (
    Dataset.from_tensor_slices((X_val, y_val))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test = (
    Dataset.from_tensor_slices((X_test, y_test))
    .batch(BATCH_SIZE)
)

inputs = layers.Input(shape=(28, 28, 1))

x = layers.Conv2D(16, 3, strides=1, padding="same", use_bias=False)(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(32, 3, strides=1, padding="same", use_bias=False)(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False)(x)
x = layers.SpatialDropout2D(0.1)(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(32, 3, strides=1, padding="same", use_bias=False)(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(32, 3, strides=1, padding="same", use_bias=False)(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(64, 3, strides=2, padding="same", use_bias=False)(x)
x = layers.SpatialDropout2D(0.1)(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(64, 3, strides=1, padding="same", use_bias=False)(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(64, 3, strides=1, padding="same", use_bias=False)(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(128, 3, strides=2, padding="same", use_bias=False)(x)
x = layers.SpatialDropout2D(0.15)(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(128, 3, strides=1, padding="same", use_bias=False)(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(128, 3, strides=1, padding="same", use_bias=False)(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dense(128, use_bias=False)(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss = CategoricalFocalCrossentropy(gamma=2.0, from_logits=False, label_smoothing=0.1),
    metrics=["accuracy"]
)

model.fit(train, validation_data=val, epochs=50, verbose=2, callbacks=[
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
])

predict = model.predict(test)
predict_labels = predict.argmax(axis=1)
y_test_labels = y_test.argmax(axis=1)
accuracy = accuracy_score(y_test_labels, predict_labels)
print("Test accuracy:", accuracy)
Test accuracy: 0.9888023990301793
