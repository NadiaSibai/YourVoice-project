import mediapipe as mp
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import os
import shutil
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import glob
import cv2
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# Mediapipe modules for holistic model and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ----------- MEDIAPIPE UTILS ----------- #
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def adjust_landmarks(arr, center):
    arr_reshaped = arr.reshape(-1, 3)
    center_repeated = np.tile(center, (len(arr_reshaped), 1))
    arr_adjusted = arr_reshaped - center_repeated
    return arr_adjusted.reshape(-1)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)

    nose = pose[:3]
    lh_wrist = lh[:3]
    rh_wrist = rh[:3]

    pose_adjusted = adjust_landmarks(pose, nose)
    lh_adjusted = adjust_landmarks(lh, lh_wrist)
    rh_adjusted = adjust_landmarks(rh, rh_wrist)

    return pose_adjusted, lh_adjusted, rh_adjusted

# ----------- DATASET CONFIG ----------- #
# List of new words to train on
new_words = [] #only specific selected words
ranges=[(120,152)]
# Ensure only new words are selected
selected_words = [word for word in new_words]
for start, end in ranges:
    selected_words.extend([str(num).zfill(4) for num in range(start, end)])
print(selected_words)


# ----------- DATA EXTRACTION ----------- #
def make_keypoint_arrays(path, signer, split):
    os.makedirs(f'karsl-502/{signer}/{split}', exist_ok=True)
    working_path = f'karsl-502/{signer}/{split}'
    words_folder = os.path.join(path, str(signer), str(signer), split)

    for word in tqdm(selected_words):
        video_files = os.listdir(os.path.join(words_folder, word))
        for video_file in video_files:
            video = sorted(os.listdir(os.path.join(words_folder, word, video_file)))
            pose_keypoints, lh_keypoints, rh_keypoints = [], [], []

            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                for frame in video:
                    frame_path = os.path.join(words_folder, word, video_file, frame)
                    frame = cv2.imread(frame_path)
                    image, results = mediapipe_detection(frame, holistic)
                    pose, lh, rh = extract_keypoints(results)
                    pose_keypoints.append(pose)
                    lh_keypoints.append(lh)
                    rh_keypoints.append(rh)

            for keypoints, name in zip([pose_keypoints, lh_keypoints, rh_keypoints],
                                       ['pose_keypoints', 'lh_keypoints', 'rh_keypoints']):
                directory = os.path.join(working_path, word, name)
                os.makedirs(directory, exist_ok=True)
                np.save(os.path.join(directory, video_file), keypoints)

# Generate keypoint arrays
make_keypoint_arrays(r'/Applications/KARSL-100/KARSL-100','01','train')
#make_keypoint_arrays(r'/Applications/KARSL-100/KARSL-100','01','test')
#make_keypoint_arrays(r'/Applications/KARSL-100/KARSL-100','02','train')
make_keypoint_arrays(r'/Applications/KARSL-100/KARSL-100','02','test')

# ----------- LABEL PREPARATION ----------- #
karsl_df = pd.read_excel(r"/Applications/KARSL-100/KARSL-100/KARSL-100_Labels.xlsx")
mask = [str(i).zfill(4) in selected_words for i in karsl_df['SignID'].values]
karsl_6 = karsl_df[mask].reset_index(drop=True)

w2id = {w: i for w, i in zip(karsl_6['Sign-Arabic'].values, karsl_6['SignID'].values)}
words = np.array([v for v in karsl_6['Sign-Arabic']])
label_map = {label: num for num, label in enumerate(words)}

# ----------- PREPROCESSING ----------- #
def pad_sequence(arr, f_avg):
    while arr.shape[0] < f_avg:
        arr = np.concatenate((arr, np.expand_dims(arr[-1, :], axis=0)), axis=0)
    return arr[:f_avg, :]

def preprocess_data(data_path, signers, split, f_avg):
    sequences, labels = [], []
    for word in tqdm(words):
        for signer in signers:
            base_path = os.path.join(data_path, str(signer), split, str(w2id[word]).zfill(4))
            for sequence in os.listdir(os.path.join(base_path, 'lh_keypoints')):
                lh = np.load(os.path.join(base_path, 'lh_keypoints', sequence))
                rh = np.load(os.path.join(base_path, 'rh_keypoints', sequence))
                pose = np.load(os.path.join(base_path, 'pose_keypoints', sequence))

                lh = pad_sequence(lh, f_avg)
                rh = pad_sequence(rh, f_avg)
                pose = pad_sequence(pose, f_avg)

                sequence_combined = np.concatenate((pose, lh, rh), axis=1)
                sequences.append(sequence_combined)
                labels.append(label_map[word])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    return X, y

# ----------- TRAIN / TEST / VAL SPLIT ----------- #
data_path = 'karsl-502'
X_train, y_train = preprocess_data(data_path, ['01'], 'train', 48)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_test, y_test = preprocess_data(data_path, ['02'], 'test', 48)

# ----------- MODEL ----------- #
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dense(32, activation='relu'),
    Dense(len(words), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True, verbose=1)

#fit
model_training_history = model.fit(
    X_train, y_train,
    batch_size=32,
    validation_data=(X_val, y_val),
    validation_batch_size=32,
    epochs=50,
    callbacks=[early_stopping]
)

# Evaluation
model.evaluate(X_train, y_train)
model_evaluation_loss, model_evaluation_accuracy = model.evaluate(X_test, y_test)

# Print Model Stats
print("\n--- Model Training Statistics ---")
print(f"Final Training Accuracy: {model_training_history.history['categorical_accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {model_training_history.history['val_categorical_accuracy'][-1]:.4f}")
print(f"Final Training Loss: {model_training_history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {model_training_history.history['val_loss'][-1]:.4f}")
print(f"Test Loss: {model_evaluation_loss:.4f}")
print(f"Test Accuracy: {model_evaluation_accuracy:.4f}")

# Classification Report
print("\n--- Classification Report on Test Set ---")
yhat_probs = model.predict(X_test)
y_true_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(yhat_probs, axis=1)
print(classification_report(y_true_labels, y_pred_labels, zero_division=0))

# Plotting

def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]
    epochs = range(len(metric_value_1))
    plt.plot(epochs, metric_value_1, 'blue', label=metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label=metric_name_2)
    plt.title(str(plot_name))
    plt.legend()
    plt.show()

plot_metric(model_training_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')

# Predictions
# res = model.predict(X_test)
# for i in range(len(res)):
#     print(f"Sample {i+1}:")
#     print("Predicted sign:", words[np.argmax(res[i])])
#     print("Actual sign:", words[np.argmax(y_test[i])])
#     print("-" * 30)

# Save model
current_date_time_string = dt.datetime.strftime(dt.datetime.now(), '%Y_%m_%d__%H_%M_%S')
model_file_name = f'Kaleem_model_2_signers___Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.keras'
model.save(model_file_name)

# Prepare for Confusion Matrix
def get_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

ytrue = y_true_labels.tolist()
yhat = y_pred_labels.tolist()
y = [get_key_by_value(label_map, v) for v in ytrue]
y = [karsl_6[karsl_6['Sign-Arabic'] == v]['Sign-English'].values[0] for v in y]
ypred = [get_key_by_value(label_map, v) for v in yhat]
ypred = [karsl_6[karsl_6['Sign-Arabic'] == v]['Sign-English'].values[0] for v in ypred]

# Confusion Matrix
y_subset = y
ypred_subset = ypred
class_labels = np.unique(y_subset)
cm = confusion_matrix(y_subset, ypred_subset, labels=class_labels)
df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'

plt.figure(figsize=(12, 10))
sns.set(font_scale=1.2)
sns.heatmap(df_cm, cmap="Blues", annot=True, fmt="d", annot_kws={"size": 10})
plt.title("Confusion Matrix - All Test Samples")
plt.show()