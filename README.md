# A-Secure-Biometric-Verification-System-Using-Artificial-Intelligence-and-Encryption-Technologies
**Biometric Access Control System with 2FA**   Developed using **Python, OpenCV, dlib, and SQLite** for secure user management. Features include:   - **Biometric identity access control**   - **Liveness detection** (wink /head motion)   - **Flexible integration** for enterprise security needs.



import numpy as np
import face_recognition
import os
import sqlite3
import requests
import dlib
import cv2
from scipy.spatial import distance as dist
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding

# --- إعدادات الأمان وقاعدة البيانات ---
DB_FILE = "persons.db"
KEY_FILE = "encryption_key.key"
IV = b'\x00' * 16  # يجب أن يكون مطابقاً لما استخدمته في كود التخزين

# 1. تحميل المفتاح
with open(KEY_FILE, "rb") as f:
    AES_KEY = f.read()

def decrypt_image_from_db(encrypted_data):
    """فك تشفير البيانات الثنائية المستخرجة من قاعدة البيانات"""
    cipher = Cipher(algorithms.AES(AES_KEY), modes.CBC(IV), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
    
    # إزالة الحشو (Unpadding)
    unpadder = padding.PKCS7(128).unpadder()
    data = unpadder.update(padded_data) + unpadder.finalize()
    return data

# 2. تحميل الصور من قاعدة البيانات بدلاً من المجلد
def load_known_faces_from_db():
    images = []
    classNames = []
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT name, encrypted_image FROM persons")
    rows = cursor.fetchall()
    
    for row in rows:
        name = row[0]
        encrypted_blob = row[1]
        
        try:
            # فك التشفير
            decrypted_data = decrypt_image_from_db(encrypted_blob)
            
            # تحويل البيانات الثنائية إلى مصفوفة صور يفهمها OpenCV
            nparr = np.frombuffer(decrypted_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                images.append(img)
                classNames.append(name)
                print(f"✅ تم تحميل وفك تشفير وجه: {name}")
        except Exception as e:
            print(f"❌ فشل فك تشفير وجه {name}: {e}")
            
    conn.close()
    return images, classNames

# --- إعدادات كشف الحيوية (Liveness) ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)
EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 3
wink_counter = 0

# --- تشغيل النظام ---
print("📂 جاري استخراج الوجوه من قاعدة البيانات المشفرة...")
loaded_images, classNames = load_known_faces_from_db()

def findEncodeings(images):
    encodeList = []
    for idx, img in enumerate(images):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img_rgb)
        if len(encodes) > 0:
            encodeList.append(encodes[0])
    return encodeList

encodeListKnown = findEncodeings(loaded_images)
print(f'🎉 تم معالجة {len(encodeListKnown)} وجه بنجاح. الكاميرا ستعمل الآن...')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success: break

    img_small = cv2.resize(img, (0,0), None, 0.25, 0.25)
    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(img_rgb)
    face_encodes = face_recognition.face_encodings(img_rgb, face_locations)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    wink_detected = False
    for rect in rects:
        shape = predictor(gray, rect)
        shape = np.array([[p.x, p.y] for p in shape.parts()])
        leftEAR = eye_aspect_ratio(shape[lStart:lEnd])
        rightEAR = eye_aspect_ratio(shape[rStart:rEnd])
        ear = (leftEAR + rightEAR) / 2.0

        if ear < EYE_AR_THRESH:
            wink_counter += 1
        else:
            if wink_counter >= EYE_AR_CONSEC_FRAMES:
                wink_detected = True
            wink_counter = 0

    for encodeface, faceloc in zip(face_encodes, face_locations):
        matches = face_recognition.compare_faces(encodeListKnown, encodeface)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeface)
        
        if len(faceDis) > 0:
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                # رسم المربع
                y1, x2, y2, x1 = [v*4 for v in faceloc]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{name} - LIVE: {wink_detected}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # إرسال البيانات للسيرفر
                status = "yesyes" if wink_detected else "yesno"
                try:
                    requests.post("http://localhost:5000/api/face_action", 
                                  json={"status": status, "name": name}, timeout=0.5)
                except:
                    pass

    cv2.imshow('Biometric Secure System', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
