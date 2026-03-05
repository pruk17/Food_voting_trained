import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import MobileNetV2

# --- 1. SET PATHS ---
# ระบุที่อยู่โฟลเดอร์รูปภาพและไฟล์ CSV ที่คุณมีอยู่ตอนนี้
IMAGE_DIR = "/home/chaiyapruk/ComVision_train/Food_voting_trained/Questionair Images"
CSV_PATH = "/home/chaiyapruk/ComVision_train/Food_voting_trained/data_from_questionaire.csv"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ตั้งค่าให้ใช้ GPU แบบประหยัด Memory (ป้องกัน Memory Error)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- 2. IMAGE LOADER ---
def load_img(img_name):
    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((224, 224, 3))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return img / 255.0

# --- 3. PREPARE DATA ---
print("Loading data...")
df = pd.read_csv(CSV_PATH)
L, R, labels = [], [], []

for _, row in df.iterrows():
    L.append(load_img(row['Image 1']))
    R.append(load_img(row['Image 2']))
    labels.append(0 if row['Winner'] == 1 else 1)

X_L, X_R, y = np.array(L), np.array(R), np.array(labels)

# --- 4. BUILD IMPROVED SIAMESE MODEL ---
# Data Augmentation Layer to fight Overfitting
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False

in_l, in_r = layers.Input((224,224,3)), layers.Input((224,224,3))
feat_l = layers.GlobalAveragePooling2D()(base(in_l))
feat_r = layers.GlobalAveragePooling2D()(base(in_r))

diff = layers.Subtract()([feat_l, feat_r])
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.02))(diff)
x = layers.Dropout(0.5)(x) 
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)

out = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=[in_l, in_r], outputs=out)
# Use a smaller Learning Rate for better convergence
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

# --- 5. START TRAINING ---
print("Starting Training Phase...")
model.fit(
    [X_L, X_R], y, 
    epochs=20, 
    batch_size=8, 
    validation_split=0.2
)

# --- 6. SAVE ---
model.save("/home/chaiyapruk/ComVision_train/Food_voting_trainedfood_ranker_improved(2).keras")
print("Training finished. Model saved.")