# ===============================
# 1. IMPORT LIBRARY
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 2. LOAD DATASET (AUTO DETECT SEPARATOR)
# ===============================
df = pd.read_csv('student-mat.csv', sep=None, engine='python')

print("=== DATA AWAL ===")
print(df.head())

print("\nJumlah data:", df.shape)

# ===============================
# 3. CEK KOLOM (PENTING)
# ===============================
print("\n=== NAMA KOLOM ===")
print(df.columns)

# Pastikan kolom G3 ada
if 'G3' not in df.columns:
    raise Exception("Kolom 'G3' tidak ditemukan! Cek kembali dataset kamu.")

# ===============================
# 4. MEMBUAT LABEL (Lulus / Tidak)
# ===============================
df['Label_Lulus'] = df['G3'].apply(lambda x: 'Lulus' if x >= 10 else 'Tidak Lulus')

print("\nDistribusi Label:")
print(df['Label_Lulus'].value_counts())

# ===============================
# 5. EKSPLORASI DATA
# ===============================
print("\n=== INFO DATA ===")
print(df.info())

print("\n=== STATISTIK DESKRIPTIF ===")
print(df.describe())

# ===============================
# 6. CEK MISSING VALUE
# ===============================
print("\n=== MISSING VALUE ===")
print(df.isnull().sum())

# ===============================
# 7. VISUALISASI
# ===============================

# Histogram G3
plt.figure()
df['G3'].hist()
plt.title('Histogram Nilai Akhir (G3)')
plt.xlabel('Nilai')
plt.ylabel('Frekuensi')
plt.show()

# Boxplot Absences
plt.figure()
plt.boxplot(df['absences'])
plt.title('Boxplot Absences')
plt.show()

# ===============================
# 8. HANDLE OUTLIER (CAPPING)
# ===============================
df['absences'] = np.where(df['absences'] > 30, 30, df['absences'])

# ===============================
# 9. ENCODING KATEGORIK (AMAN)
# ===============================
for col in df.select_dtypes(include='object').columns:
    unique_vals = df[col].dropna().unique()
    
    # hanya encode kalau isinya yes/no
    if set(unique_vals).issubset({'yes', 'no'}):
        df[col] = df[col].map({'yes': 1, 'no': 0})

# ===============================
# 10. NORMALISASI (MIN-MAX)
# ===============================
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    min_val = df[col].min()
    max_val = df[col].max()
    
    if max_val != min_val:  # hindari pembagian nol
        df[col] = (df[col] - min_val) / (max_val - min_val)

# ===============================
# 11. FEATURE ENGINEERING
# ===============================
df['Rata_Nilai'] = (df['G1'] + df['G2']) / 2

# ===============================
# 12. HASIL AKHIR
# ===============================
print("\n=== DATA SETELAH PREPROCESSING ===")
print(df.head())

print("\nJumlah data akhir:", df.shape)

# ===============================
# 13. SIMPAN DATA
# ===============================
df.to_csv('student_processed.csv', index=False)

print("\nDataset berhasil disimpan sebagai 'student_processed.csv'")