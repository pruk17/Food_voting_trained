import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os

# --- ตั้งค่า Path สำหรับวันทดสอบ ---
#pip install nvidia-cudnn-cu12
TEST_IMAGE_DIR = "/home/chaiyapruk/ComVision_train/Food_voting_trained/Questionair Images" 
TEST_CSV_PATH = "/home/chaiyapruk/ComVision_train/Food_voting_trained/data_from_questionaire.csv"
MODEL_PATH = "/home/chaiyapruk/ComVision_train/Food_voting_trained/food_ranker_improved(1).keras"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
# 1. โหลดโมเดลที่คุณเทรนไว้ [cite: 47]
model = tf.keras.models.load_model(MODEL_PATH)

def load_img(img_name):
    img_path = os.path.join(TEST_IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None: return np.zeros((224, 224, 3))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return img / 255.0

# 2. อ่านไฟล์ test.csv และทำนายผล [cite: 36, 44]
df_test = pd.read_csv(TEST_CSV_PATH)
results = []

print("Predicting... please wait.")
for _, row in df_test.iterrows():
    img1 = load_img(row['Image 1'])
    img2 = load_img(row['Image 2'])
    
    # ส่งภาพคู่เข้าไปในโมเดล
    pred = model.predict([np.expand_dims(img1, 0), np.expand_dims(img2, 0)], verbose=0)
    
    # ถ้าค่า > 0.5 ให้ตอบ 2 (ภาพขวาชนะ), ถ้า < 0.5 ให้ตอบ 1 (ภาพซ้ายชนะ) [cite: 4, 45]
    results.append(2 if pred[0][0] > 0.5 else 1)

# 3. บันทึกผลลงไฟล์ใหม่เพื่อส่งงาน [cite: 44]
df_test['Winner'] = results
df_test.to_csv("test_result(1).csv", index=False)
print("Saved result to test_result.csv")