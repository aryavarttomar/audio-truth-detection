Truth Detection from Audio: ML-Powered Story Classification

This project aims to classify 30-second narrated audio stories as either true or false using machine learning. Leveraging the MLEnd Deception Dataset, the system extracts meaningful audio features and applies supervised learning techniques to detect deception in speech.

Project Objective

To develop a machine learning pipeline that can identify truthfulness from audio data using acoustic features such as MFCCs, pitch, power, and spectral characteristics.

Dataset

MLEnd Deception Dataset

Contains short audio clips labeled as true or false

Includes language metadata for each sample

Pipeline Overview

Preprocessing:

Standardized all audio to 44,100 Hz

Binary label encoding (1 for true, 0 for false)

Stratified train-test split

Audio Chunking:

Each file is segmented into: First 30s, Middle 30s, Last 30s

Feature Extraction:

MFCCs (Mel Frequency Cepstral Coefficients)

Spectral features (centroid, bandwidth, rolloff)

Pitch statistics and voiced frame rate

Power (RMS energy)

Model Training:

SVM, KNN, and Logistic Regression

Evaluated with accuracy, precision, recall, F1-score

Ensemble Learning:

Combined SVM and KNN using hard voting

Best results achieved with the middle audio chunk (up to 60% test accuracy)

Key Findings

Middle chunk of audio was most predictive

Language alone did not significantly influence truth classification

Power and MFCC_1 were the most important features

Ensemble models outperformed individual models

Future Improvements

Explore additional emotional or prosodic features

Try advanced ensembles like soft voting or stacking

Address class imbalance

Experiment with overlapping or variable-length chunking
