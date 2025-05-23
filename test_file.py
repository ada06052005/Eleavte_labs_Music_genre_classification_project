# STEP 1: IMPORT LIBRARIES
import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import joblib

# STEP 2: FEATURE EXTRACTION FUNCTION
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, duration=30)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        return None

# STEP 3: DATA PREPARATION
data_path = "C:/Users/Harshada/Downloads/genres"
genres = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
X = []
y = []

for genre in genres:
    genre_path = os.path.join(data_path, genre)
    for file in os.listdir(genre_path):
        if file.endswith(".au"):
            file_path = os.path.join(genre_path, file)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(genre)

X = np.array(X)
y = np.array(y)

# STEP 4: ENCODING LABELS
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)
num_genres = y_categorical.shape[1]  # Ensures correct number of output classes

# STEP 5: TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# STEP 6: MODEL BUILDING
model = Sequential()
model.add(Dense(256, input_shape=(13,), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_genres, activation='softmax'))  # Uses dynamic genre count

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# STEP 7: TRAIN THE MODEL
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# STEP 8: EVALUATE THE MODEL
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))

# STEP 9: SAVE MODEL AND LABEL ENCODER
model.save("genre_classification_model.h5")
joblib.dump(label_encoder, "label_encoder.pkl")

# STEP 10: GENRE PREDICTION FUNCTION
def predict_genre(file_path, model, label_encoder):
    features = extract_features(file_path)
    if features is not None:
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction)
        predicted_genre = label_encoder.inverse_transform([predicted_index])[0]
        return predicted_genre
    else:
        return "Could not extract features"

# STEP 11: PREDICT FOR MULTIPLE TEST FILES
test_folder = test_folder = r"C:\Users\Harshada\Downloads\test_sample"
print("\n--- PREDICTIONS FOR TEST FILES ---")
for test_file in os.listdir(test_folder):
    if test_file.endswith(".au"):
        test_path = os.path.join(test_folder, test_file)
        predicted = predict_genre(test_path, model, label_encoder)
        print(f"{test_file} -> Predicted genre: {predicted}")
