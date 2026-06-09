# ==========================================
# IMPORT LIBRARY
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================
# MEMBACA DATASET
# ==========================================

print("MEMBACA DATASET...")

df = pd.read_csv("student_processed.csv")

print("\n5 DATA PERTAMA")
print(df.head())

# ==========================================
# ENCODING DATA KATEGORIKAL
# ==========================================

print("\nMELAKUKAN ENCODING DATA...")

label_encoder = LabelEncoder()

# mencari atribut kategorikal
categorical_columns = df.select_dtypes(include=['object']).columns

# melakukan encoding
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

print("Encoding selesai")

# ==========================================
# MEMISAHKAN FITUR DAN TARGET
# ==========================================

print("\nMEMISAHKAN FITUR DAN TARGET...")

# G3 DIHAPUS UNTUK MENGHINDARI DATA LEAKAGE
X = df.drop(["Label_Lulus", "G3"], axis=1)

# target
y = df["Label_Lulus"]

print("Jumlah fitur :", X.shape[1])

# ==========================================
# DATA SPLITTING
# ==========================================

print("\nMELAKUKAN DATA SPLITTING...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Jumlah Data Training :", len(X_train))
print("Jumlah Data Testing  :", len(X_test))

# ==========================================
# BASELINE DECISION TREE
# ==========================================

print("\nTRAINING BASELINE DECISION TREE...")

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(X_train, y_train)

print("Decision Tree berhasil dilatih")

# ==========================================
# BASELINE NAIVE BAYES
# ==========================================

print("\nTRAINING BASELINE NAIVE BAYES...")

nb_model = GaussianNB()

nb_model.fit(X_train, y_train)

print("Naive Bayes berhasil dilatih")

# ==========================================
# PREDIKSI BASELINE MODEL
# ==========================================

print("\nMELAKUKAN PREDIKSI BASELINE MODEL...")

dt_pred = dt_model.predict(X_test)

nb_pred = nb_model.predict(X_test)

# ==========================================
# EVALUASI BASELINE MODEL
# ==========================================

print("\nHASIL EVALUASI BASELINE MODEL")

# accuracy decision tree
dt_acc = accuracy_score(y_test, dt_pred)

# accuracy naive bayes
nb_acc = accuracy_score(y_test, nb_pred)

print("\nAccuracy Decision Tree :", dt_acc)
print("Accuracy Naive Bayes   :", nb_acc)

# ==========================================
# PERBANDINGAN BASELINE MODEL
# ==========================================

print("\nPERBANDINGAN BASELINE MODEL")

if dt_acc > nb_acc:
    print("Decision Tree sementara lebih unggul")

elif nb_acc > dt_acc:
    print("Naive Bayes sementara lebih unggul")

else:
    print("Kedua model memiliki accuracy yang sama")

# ==========================================
# HYPERPARAMETER TUNING
# ==========================================

print("\nMELAKUKAN HYPERPARAMETER TUNING...")

param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)

# training tuning
grid_search.fit(X_train, y_train)

# model terbaik
best_dt_model = grid_search.best_estimator_

# prediksi tuning
best_pred = best_dt_model.predict(X_test)

# accuracy tuning
best_acc = accuracy_score(y_test, best_pred)

print("\nHASIL HYPERPARAMETER TUNING")

print("Best Parameter :", grid_search.best_params_)

print("Best Accuracy  :", best_acc)

# ==========================================
# EVALUASI LANJUTAN
# ==========================================

print("\nEVALUASI LANJUTAN MODEL")

# confusion matrix
cm = confusion_matrix(y_test, best_pred)

# precision
precision = precision_score(y_test, best_pred)

# recall
recall = recall_score(y_test, best_pred)

# f1-score
f1 = f1_score(y_test, best_pred)

print("\nCONFUSION MATRIX")
print(cm)

print("\nPrecision :", precision)

print("Recall    :", recall)

print("F1-Score  :", f1)

# ==========================================
# VISUALISASI DECISION TREE
# ==========================================

print("\nMENAMPILKAN VISUALISASI DECISION TREE...")

plt.figure(figsize=(18,8))

plot_tree(
    best_dt_model,
    filled=True,
    feature_names=X.columns,
    class_names=["Tidak Lulus", "Lulus"],
    rounded=True
)

plt.title("Visualisasi Decision Tree Setelah Hyperparameter Tuning")

plt.show()

# ==========================================
# INFORMASI ROOT NODE
# ==========================================

print("\nINFORMASI ROOT NODE")

print(
    "Root Node :",
    X.columns[best_dt_model.tree_.feature[0]]
)

# ==========================================
# PROGRAM SELESAI
# ==========================================

print("\nPROGRAM SELESAI")
