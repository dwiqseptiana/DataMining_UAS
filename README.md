# DataMining_UAS
Repository ini berisi proyek Data Mining end-to-end menggunakan Student Performance Dataset, meliputi preprocessing data, exploratory data analysis (EDA), baseline modeling, hyperparameter tuning Decision Tree, evaluasi model klasifikasi.

Student Performance Classification using Data Mining

Deskripsi Proyek
Repository ini berisi proyek Data Mining end-to-end menggunakan Student Performance Dataset. Proyek dilakukan sebagai bagian dari tugas UTS, Tugas 2, dan UAS mata kuliah Data Mining dengan pendekatan bertahap mulai dari preprocessing data hingga evaluasi model klasifikasi.
Penelitian berfokus pada prediksi status kelulusan siswa menggunakan algoritma machine learning seperti Decision Tree dan Naive Bayes.

Dataset
Dataset yang digunakan:
Student Performance Dataset
Sumber: Kaggle / UCI Machine Learning Repository

Dataset:
https://www.kaggle.com/datasets/uciml/student-alcohol-consumption
Jumlah data:
395 baris data
35 atribut

Tahapan Proyek
1. Exploratory Data Analysis (EDA)
   
Tahapan eksplorasi data meliputi:
Analisis struktur dataset
Statistik deskriptif
Distribusi target
Visualisasi data
Identifikasi missing value dan outlier

Visualisasi yang digunakan:
Histogram
Boxplot

2. Preprocessing Data

Tahapan preprocessing meliputi:
Pembuatan label klasifikasi
Handling outlier
Encoding data kategorikal
Normalisasi menggunakan Min-Max Scaling
Feature engineering
Penyimpanan dataset hasil preprocessing

Output preprocessing:
student_processed.csv

3. Baseline Modeling

Model baseline yang digunakan:
Decision Tree
Naive Bayes

Tahapan:
Data splitting
Training model
Evaluasi accuracy awal
Visualisasi Decision Tree

4. Hyperparameter Tuning

Optimasi model dilakukan menggunakan:
GridSearchCV

Parameter yang diuji:
max_depth
min_samples_split
min_samples_leaf

5. Evaluasi Model

Evaluasi dilakukan menggunakan:
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
