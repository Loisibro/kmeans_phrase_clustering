from utils import load_data

df = load_data("data/phrase_clustering_dataset.json")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from utils import load_data

# Step 1: Load the dataset
df = load_data("data/phrase_clustering_dataset.json")

# Step 2: Check what's inside (adjust column names as needed)
print("Sample data:")
print(df.head())

# Step 3: Extract phrases (adjust the key if different)
# Flatten the nested dictionary into a DataFrame of phrases and categories
flattened_phrases = []

for group in df.index:
    for category in df.columns:
        phrase_lists = df.loc[group, category]
        if isinstance(phrase_lists, list):  # Only proceed if it's a list
            for phrase_group in phrase_lists:
                if isinstance(phrase_group, list):
                    for phrase in phrase_group:
                        flattened_phrases.append((phrase, category))

# Create a new DataFrame
flat_df = pd.DataFrame(flattened_phrases, columns=["phrase", "category"])

# Now use the 'phrase' column
phrases = flat_df["phrase"]

# Step 4: Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(phrases)

# Step 5: Apply K-Means clustering
k = 5  # You can try different values
model = KMeans(n_clusters=k, random_state=42)
model.fit(X)
# Step: Combine phrases and categories with cluster labels
clustered_df = pd.DataFrame(flattened_phrases, columns=["phrase", "category"])
clustered_df["cluster"] = model.labels_

# Print sample clustered output
print("Clustered sample:")
print(clustered_df.head())

# Optional: Save to file for review
clustered_df.to_csv("clustered_phrases.csv", index=False)


# Step 7: Display results
for i in range(k):
    print(f"\nCluster {i}:")
    print(clustered_df[clustered_df["cluster"] == i]["phrase"].head(5).to_string(index=False))

# Optional: Visualize using PCA (2D)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X.toarray())

plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=model.labels_, cmap='rainbow')
plt.title("K-Means Clusters (2D Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
# Export Clustered Data to CSV
clustered_df.to_csv("clustered_phrases.csv", index=False)
