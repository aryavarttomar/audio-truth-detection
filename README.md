# Truth Detection from Audio: ML-Powered Story Classification

This project focuses on building a machine learning pipeline to classify **30-second audio clips** of narrated stories as **true** or **false**. Using the [MLEnd Deception Dataset](https://github.com/declare-lab/MLEND), we extract acoustic features and train multiple ML models to detect deception from voice recordings.

---

##  Project Objective

To develop a robust system that leverages audio features and machine learning techniques to determine the veracity of narrated stories. This has practical applications in fields such as **forensic linguistics**, **content moderation**, and **deception detection**.

---

##  Dataset

**MLEnd Deception Dataset**  
- Contains short audio recordings of narrated stories labeled as `true` or `false`
- Includes metadata such as language

🔗 Dataset: [https://github.com/declare-lab/MLEND](https://github.com/declare-lab/MLEND)

---

##  Pipeline Overview

### 1. **Preprocessing**
- Standardized sampling rate (44,100 Hz)
- Labeled true/false stories (1/0)
- Stratified train-test split with language grouping

### 2. **Audio Chunking**
- Each audio file is segmented into:
  - First 30 seconds
  - Middle 30 seconds
  - Last 30 seconds

### 3. **Feature Extraction** (using `librosa`)
- MFCCs
- Pitch (mean, std)
- Power (RMS energy)
- Spectral features (centroid, bandwidth, rolloff)
- Voiced Frame Rate

### 4. **Model Training**
- Models used:
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix

### 5. **Ensemble Learning**
- **Hard Voting** ensemble (SVM + KNN) outperformed individual models
- Best results from **Middle Chunk** (~60% test accuracy)

---

##  Key Findings

- **Middle Chunk** was most predictive of truth/deception.
- Language alone had **no significant effect** on classification accuracy.
- **Power** and **MFCC_1** were the most correlated features with the label.
- Ensemble model improved classification performance over individual models.

---

##  Future Improvements

- Test advanced ensemble techniques (e.g., soft voting, stacking)
- Explore additional prosodic or emotional features
- Address class imbalance using resampling or class weighting
- Evaluate chunk fusion strategies or overlapping windows

---

##  Installation & Usage

```bash
# Clone the repository
git clone https://github.com/your-username/audio-truth-detection.git
cd audio-truth-detection

# Install dependencies
pip install -r requirements.txt

# Run feature extraction
python extract_features.py

# Train models and evaluate
python train_models.py
