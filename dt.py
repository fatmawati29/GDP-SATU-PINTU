from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd

# Data latihan
data = {
    "Skor": [3, 2, 1, 0],
    "Salah_GDP": [1, 0, 0, 1],
    "Salah_NNP": [0, 1, 0, 1],
    "Salah_NNI": [0, 0, 1, 1],
    "Rekomendasi": ["Pelajari GDP", "Pelajari NNP", "Pelajari NNI", "Pelajari Semua"]
}

df = pd.DataFrame(data)

# Fitur (input) dan target (output)
X = df[["Skor", "Salah_GDP", "Salah_NNP", "Salah_NNI"]]
y = df["Rekomendasi"]

# Buat model Decision Tree
model = DecisionTreeClassifier()
model.fit(X, y)

# Visualisasi Decision Tree (opsional)
tree.plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)

# Contoh data siswa
new_data = pd.DataFrame({
    "Skor": [2],
    "Salah_GDP": [0],
    "Salah_NNP": [1],
    "Salah_NNI": [0]
})

# Prediksi rekomendasi
recommendation = model.predict(new_data)
print(f"Rekomendasi: {recommendation[0]}")
