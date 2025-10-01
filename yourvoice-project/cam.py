import cv2
import numpy as np
from collections import deque
from keras.models import load_model
import mediapipe as mp
from bidi.algorithm import get_display
import arabic_reshaper
import pandas as pd
from PIL import ImageFont, ImageDraw, Image

# Load model
model = load_model(r"/Users/nadiasibai/Desktop/CS316/Project-run/ai-project-try/Kaleem_model_2_signers___Date_Time_2025_04_29__00_42_47___Loss_1.5047756433486938___Accuracy_0.64453125-newes.keras ")

# Load label mappings
karsl_df = pd.read_excel("/Users/nadiasibai/Desktop/CS316/Project-run/KARSL-100_Labels\ \(1).xlsx")
ranges = [(120, 155)]
selected_words = [str(num).zfill(4) for start, end in ranges for num in range(start, end)]
karsl_6 = karsl_df[karsl_df['SignID'].astype(str).str.zfill(4).isin(selected_words)].reset_index(drop=True)
label_map = {label: idx for idx, label in enumerate(karsl_6['Sign-Arabic'])}
reverse_label_map = {v: k for k, v in label_map.items()}

# Mediapipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def adjust_landmarks(arr, center):
    arr_reshaped = arr.reshape(-1, 3)
    center_repeated = np.tile(center, (len(arr_reshaped), 1))
    return (arr_reshaped - center_repeated).reshape(-1)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))

    pose_center = pose[0] if pose.any() else np.zeros(3)
    lh_center = lh[0] if lh.any() else np.zeros(3)
    rh_center = rh[0] if rh.any() else np.zeros(3)

    pose = adjust_landmarks(pose.flatten(), pose_center)
    lh = adjust_landmarks(lh.flatten(), lh_center)
    rh = adjust_landmarks(rh.flatten(), rh_center)

    return pose, lh, rh

# Setup Arabic font
font_path = "arial.ttf"  # Update with a full path to an Arabic-supporting font
font = ImageFont.truetype(font_path, 32)

def draw_arabic_text(img, text, position):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    draw.text(position, bidi_text, font=font, fill=(255, 255, 255))
    return np.array(img_pil)

# Buffers
sequence_buffer = deque(maxlen=48)
prediction_buffer = deque(maxlen=15)
predicted_text = ""

# Define hand capture box area (you can tweak this size)
box_x1, box_y1, box_x2, box_y2 = 100, 100, 540, 380

# Camera & detection loop
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image, results = mediapipe_detection(frame, holistic)
        pose, lh, rh = extract_keypoints(results)
        features = np.concatenate([pose, lh, rh])
        sequence_buffer.append(features)

        draw_landmarks(image, results)

        # Draw hand guide box
        cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 0), 2)
        image = draw_arabic_text(image, "ضع يديك هنا", (box_x1 + 10, box_y1 - 40))

        if len(sequence_buffer) == 48:
            input_data = np.expand_dims(sequence_buffer, axis=0)
            prediction = model.predict(input_data, verbose=0)
            pred_class = np.argmax(prediction)
            confidence = prediction[0][pred_class]
            prediction_buffer.append(pred_class)

            most_common = max(set(prediction_buffer), key=prediction_buffer.count)

            # Filter only if prediction is confident and stable
            if prediction_buffer.count(most_common) > 10 and confidence > 0.8:
                pred_class = most_common
                arabic_word = reverse_label_map.get(pred_class, "")
                predicted_text = f"{arabic_word} ({confidence * 100:.1f}%)"

        # Draw predicted text at top
        image = cv2.rectangle(image, (0, 0), (640, 50), (0, 0, 0), -1)
        if predicted_text:
            image = draw_arabic_text(image, predicted_text, (10, 10))

        cv2.imshow("Sign Language Translator", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()