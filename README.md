# 🗣️ Speech Command Recognition using HMM

This project implements a simple **speech command recognition system** using **Hidden Markov Models (HMM)** and **MFCC features**. It classifies short spoken words such as **"yes", "no", "up", "down", "left", "right", "stop", and "go"** from the [Mini Speech Commands dataset](https://www.tensorflow.org/datasets/catalog/mini_speech_commands).

---

## 📁 Dataset

We use the `mini_speech_commands` dataset, a simplified version of the [Speech Commands dataset](https://arxiv.org/abs/1804.03209) by Google. It contains short audio samples of 8 spoken English words:

> `yes`, `no`, `up`, `down`, `left`, `right`, `stop`, `go`

---

## 🧠 How it Works

### 🔹 Feature Extraction
- Extracts **MFCC** (Mel-frequency cepstral coefficients) from `.wav` files using **Librosa**.

### 🔹 Model Training
- Trains a **Gaussian Hidden Markov Model (HMM)** for each word using **hmmlearn**.

### 🔹 Prediction
- Scores the MFCCs against each word’s HMM.
- Selects the word with the **highest log-likelihood**.

### 🔹 Evaluation
- Splits data using `train_test_split`.
- Computes accuracy using `accuracy_score` from `sklearn`.

---

## 📊 Example Features

- Each audio clip is converted into a **matrix of MFCCs**:
  - **Shape**: `(T, n_mfcc)`
  - `T` varies depending on audio duration
  - `n_mfcc = 13` (by default)

---

## 🤖 Model Details

| Property              | Value           |
|-----------------------|-----------------|
| Model Type            | Gaussian HMM    |
| Number of States      | 5 (`n_components`) |
| Covariance Type       | Diagonal        |
| Trained per word?     | ✅ Yes           |

---

## 📈 Evaluation Metric

- **Accuracy Score** using `sklearn.metrics.accuracy_score`

---

## 💻 Requirements

Install the dependencies using pip:

```bash
pip install numpy librosa hmmlearn scikit-learn
