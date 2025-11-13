# === Étape 0 : Installer/importer les bibliothèques ===
!pip install pandas seaborn matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Étape 1 : Charger la base de données ===
# D’après la page UCI, un fichier CSV “COREVQA_data.csv” est disponible
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/1198/COREVQA_data.csv"
df = pd.read_csv(url)

# === Étape 2 : Aperçu rapide ===
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe(include='all'))

# === Étape 3 : Statistiques descriptives selon types de variables ===
# Variables numériques
num_vars = df.select_dtypes(include=['float64','int64']).columns.tolist()
print("Variables numériques:", num_vars)
print(df[num_vars].describe())

# Variables catégorielles/object
cat_vars = df.select_dtypes(include=['object','category']).columns.tolist()
print("Variables catégorielles:", cat_vars)
for v in cat_vars:
    print(f"--- Variable {v} ---")
    print(df[v].value_counts(dropna=False).head(10))
    print()

# === Étape 4 : Analyse des distributions graphiques ===
# Histogrammes pour numériques
df[num_vars].hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# Boxplots pour numériques
plt.figure(figsize=(12,6))
sns.boxplot(data=df[num_vars])
plt.xticks(rotation=45)
plt.title("Boxplots – variables numériques")
plt.show()

# Pour chaque variable catégorielle, bar-plot
for v in cat_vars:
    plt.figure(figsize=(8,4))
    sns.countplot(y=df[v], order=df[v].value_counts().index[:10])
    plt.title(f"Fréquence des catégories pour {v}")
    plt.tight_layout()
    plt.show()

# === Étape 5 : Corrélations (si pertinent) ===
if len(num_vars) > 1:
    plt.figure(figsize=(10,8))
    sns.heatmap(df[num_vars].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Matrice de corrélation – variables numériques")
    plt.show()

# === Étape 6 : Valeurs manquantes & distribution de complétude ===
print("Valeurs manquantes par variable:")
print(df.isna().sum().sort_values(ascending=False).head(10))

plt.figure(figsize=(8,4))
sns.heatmap(df.isna(), cbar=False)
plt.title("Heatmap des valeurs manquantes")
plt.show()
