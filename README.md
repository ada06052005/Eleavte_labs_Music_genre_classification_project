# 🎵 Music Genre Classification Project

## 📌 Overview
This project aims to classify music files into one of 10 genres using deep learning and audio signal processing techniques. The system extracts features from `.au` audio files and predicts the most likely genre using a trained neural network model.

## 🎼 Dataset
The dataset used in this project is the **GTZAN Modified Music Genre Classification Dataset** available on Kaggle.

🔗 **Dataset Source**:  
[https://www.kaggle.com/datasets/gabrielopecs/gtzan-modified-music-genre-classification](https://www.kaggle.com/datasets/gabrielopecs/gtzan-modified-music-genre-classification)

- The dataset contains 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.
- Each genre folder contains 100 `.au` audio files (30 seconds each).
- Total files: 1000

## 🛠️ Tools and Technologies Used
- **Python 3.12**
- **Visual Studio Code** (IDE)
- **Libraries**:
  - `Librosa` – for audio processing and feature extraction
  - `NumPy`, `os` – for numerical operations and file handling
  - `scikit-learn` – for label encoding, data splitting, and evaluation
  - `TensorFlow/Keras` – for building and training the neural network
  - `Joblib` – for saving the label encoder

## ⚙️ How It Works
1. **Feature Extraction**:  
   Extracted MFCC (Mel-Frequency Cepstral Coefficients) features from audio files using `librosa`.

2. **Data Preparation**:  
   Prepared datasets by organizing features and labels, and encoded labels into numerical format.

3. **Model Architecture**:  
   A deep neural network was built using `Keras`, consisting of:
   - Dense layers with ReLU activation
   - Dropout layers for regularization
   - Final softmax layer for multi-class classification

4. **Model Training**:  
   Trained using 80% of the data and validated on the remaining 20%. Loss function used was `categorical_crossentropy`, and optimizer was `adam`.

5. **Model Evaluation**:  
   Accuracy and classification report were generated on the test set to evaluate performance.

6. **Genre Prediction**:  
   A function was created to predict the genre of any given `.au` audio file. The model was tested with a few random files stored in a separate `test_samples` folder.


