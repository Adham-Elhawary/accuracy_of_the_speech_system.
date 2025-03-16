import os

os.environ["LOKY_MAX_CPU_COUNT"] = "5"

import numpy as np
import librosa
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset_path = r"C:\Users\Adham\Downloads\mini_speech_commands\mini_speech_commands"
words = ["yes", "no", "up", "down", "left", "right", "stop", "go"]
n_mfcc = 13
n_components = 5

def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T

def load_dataset(dataset_path, words):
    data = []
    labels = []
    for word in words:
        word_path = os.path.join(dataset_path, word)
        for file_name in os.listdir(word_path):
            file_path = os.path.join(word_path, file_name)
            mfcc = extract_mfcc(file_path, n_mfcc)
            data.append(mfcc)
            labels.append(word)
    return data, labels

def train_hmm_models(data, labels, words, n_components=5):
    models = {}
    for word in words:
        word_data = [data[i] for i in range(len(data)) if labels[i] == word]
        word_data = np.vstack(word_data)
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100)
        model.fit(word_data)
        models[word] = model
    return models

def predict_word(models, mfcc):
    best_score = -float("inf")
    best_word = None
    for word, model in models.items():
        score = model.score(mfcc)
        if score > best_score:
            best_score = score
            best_word = word
    return best_word

#################################################################################################################################################
def main():
    data, labels = load_dataset(dataset_path, words)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    models = train_hmm_models(X_train, y_train, words, n_components)
    y_pred = []
    for mfcc in X_test:
        y_pred.append(predict_word(models, mfcc))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

#################################################################################################################################################
if __name__ == "__main__":
    main()