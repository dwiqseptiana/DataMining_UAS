# ==========================================
# IMPORT LIBRARY
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# ==========================================
# MEMBACA DATASET
# ==========================================

print("MEMBACA DATASET...")

df = pd.read_csv("student_processed.csv")

print("\n5 Data Pertama:")
print(df.head())

# ==========================================
# ENCODING DATA KATEGORIKAL
# ==========================================

print("\nMELAKUKAN ENCODING DATA...")

label_encoder = LabelEncoder()

# mencari semua kolom kategorikal
categorical_columns = df.select_dtypes(include=['object']).columns

# melakukan encoding
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

print("Encoding selesai")

# ==========================================
# MEMISAHKAN FITUR DAN TARGET
# ==========================================

print("\nMEMISAHKAN FITUR DAN TARGET...")

X = df.drop("Label_Lulus", axis=1)
y = df["Label_Lulus"]

print("Jumlah fitur:", X.shape[1])

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
# TRAINING DECISION TREE
# ==========================================

print("\nTRAINING DECISION TREE...")

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(X_train, y_train)

print("Decision Tree berhasil dilatih")

# ==========================================
# TRAINING NAIVE BAYES
# ==========================================

print("\nTRAINING NAIVE BAYES...")

nb_model = GaussianNB()

nb_model.fit(X_train, y_train)

print("Naive Bayes berhasil dilatih")

# ==========================================
# PREDIKSI MODEL
# ==========================================

print("\nMELAKUKAN PREDIKSI...")

dt_pred = dt_model.predict(X_test)
nb_pred = nb_model.predict(X_test)

# ==========================================
# EVALUASI MODEL
# ==========================================

print("\nHASIL EVALUASI MODEL")

# accuracy decision tree
dt_acc = accuracy_score(y_test, dt_pred)

# accuracy naive bayes
nb_acc = accuracy_score(y_test, nb_pred)

print("\nAccuracy Decision Tree :", dt_acc)
print("Accuracy Naive Bayes   :", nb_acc)

# ==========================================
# PERBANDINGAN MODEL
# ==========================================

print("\nPERBANDINGAN MODEL")

if dt_acc > nb_acc:
    print("Decision Tree sementara lebih unggul")
elif nb_acc > dt_acc:
    print("Naive Bayes sementara lebih unggul")
else:
    print("Kedua model memiliki accuracy yang sama")

# ==========================================
# VISUALISASI DECISION TREE
# ==========================================

print("\nMENAMPILKAN VISUALISASI DECISION TREE...")

plt.figure(figsize=(10,5))

plot_tree(
    dt_model,
    filled=True,
    feature_names=X.columns,
    class_names=["Tidak Lulus", "Lulus"],
    rounded=True
)

plt.title("Visualisasi Decision Tree")

plt.show()

# ==========================================
# INFORMASI ROOT NODE
# ==========================================

print("\nINFORMASI ROOT NODE")

print("Root Node:", X.columns[dt_model.tree_.feature[0]])

# ==========================================
# SELESAI
# ==========================================

print("\nPROGRAM SELESAI")