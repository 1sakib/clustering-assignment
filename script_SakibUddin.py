import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Preprocessing imports
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Clustering imports
from sklearn.cluster import KMeans, AgglomerativeClustering

#Cluster validity imports
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# CONSTANTS
RANDOM_STATE = 4
K_MIN = 2
K_MAX = 21
# -------------------------------------------------------------
# Methods : Auto K Elbow Selector

def best_k_elbow(X, k_min=K_MIN, k_max=K_MAX):
    inertias = []
    for k in range (k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto').fit(X)
        inertias.append(km.inertia_)

    deltas = np.diff(inertias)
    second_derivative = np.diff(deltas) # Calculate deriviate

    best_k = np.argmax(second_derivative) + k_min + 1

    return best_k, inertias

# -------------------------------------------------------------
# LOAD DATA

df = pd.read_csv("e-shop clothing 2008.csv", sep=';')

print("\n Data Info:")
print(df.info())

print("\n Check if there are missing values in each column:")
print(df.isna().sum()) # This yields no missing values, so we do not
                       # need to handle them.


# -------------------------------------------------------------
# PREPROCESSING


useless_columns = ["year", "day", "session ID", "order", "month"]

print("\nDataframe before preprocessing:", df.shape)
df = df.drop(columns=useless_columns)

df = df.drop_duplicates()

# Frequency encoding for "page 2 (clothing model)" column
if "page 2 (clothing model)" in df.columns:
    freq = df["page 2 (clothing model)"].value_counts()
    df["page_2_freq"] = df["page 2 (clothing model)"].map(freq)
    df = df.drop(columns=["page 2 (clothing model)"])

categorical_features = [
    "country",
    "page 1 (main category)",
    "colour",
    "location",
    "model photography",
    "page",
]

numeric_features = [
    "price",
    "price 2",
    "page_2_freq",
]

# Scaling numeric features and One-Hot Encoding categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

preprocessed_df = preprocessor.fit_transform(df)
print("\nDataframe after preprocessing:", preprocessed_df.shape)



# --------------------------------------------------------------
# CLUSTERING

best_k, inertias = best_k_elbow(preprocessed_df, k_min=K_MIN, k_max=K_MAX)
print("The best k determined by Elbow method is:", best_k)


plt.figure()
plt.plot(range(K_MIN, K_MAX + 1), inertias, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method Graph")
plt.show()

optimal_k = best_k  


# KMeans Clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
km_labels = kmeans.fit_predict(preprocessed_df)

# Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
hc_labels = hc.fit_predict(preprocessed_df.toarray())



# --------------------------------------------------------------
# CLUSTER VALIDITY


# Silhouette Score
sil_km = silhouette_score(preprocessed_df, km_labels)
sil_hc = silhouette_score(preprocessed_df, hc_labels)

print("\nSilhouette Score (KMeans):", sil_km)
print("Silhouette Score (Hierarchical):", sil_hc)

# KMeans scored higher in Silhouette Score, 
# so we will proceed with KMeans for further analysis.

# PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(preprocessed_df.toarray())

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, cmap='jet', s=11)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("KMeans PCA Visualization")
plt.show()



# ---------------------------------------------------------------
# CLUSTER INTERPRETATION

df["cluster"] = km_labels

print(df["cluster"].value_counts()) # Cluster sizes

print(df.groupby("cluster")[numeric_features].mean()) # Numeric features per cluster

for col in categorical_features:  # Categorical features per cluster
    print(f"\n{col} distribution per cluster:")
    vc = df.groupby("cluster")[col].value_counts(normalize=True)
    for cluster in sorted(df["cluster"].unique()):
        print(f"\nCluster {cluster}:")
        clustervalues = vc.loc[cluster].nlargest(5)
        print(clustervalues)