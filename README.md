# K-Means Phrase Clustering Application

## 📌 Project Overview
This project implements a **K-Means clustering algorithm** on a high-dimensional phrase dataset to identify and group semantically similar phrases. It is designed to demonstrate the application of unsupervised machine learning techniques using real-world data from the AWS Open Registry.

## 💾 Dataset
**Source:** Phrase Clustering Dataset (PCD)  
**Format:** Nested JSON structure with grouped phrases under various product categories.

- The dataset was manually saved and placed under the `/data` folder.
- Example categories: `Arts_Crafts_and_Sewing`, `Home_Improvement`, `Patio_Lawn_and_Garden`, etc.

## ⚙️ Project Structure

```

kmeans\_phrase\_clustering/
│
├── data/
│   └── phrase\_clustering\_dataset.json     # Input dataset
│
├── main.py                                # Main clustering application
├── utils.py                               # Data loading and preprocessing functions
├── requirements.txt                       # (Optional) List of dependencies
└── README.md                              # Project overview

```

## 🧠 Features & Workflow

1. **Data Loading**: JSON data is read and normalized into a flat structure using `pandas`.
2. **Phrase Flattening**: Extracts individual phrases with their corresponding categories.
3. **Text Vectorization**: Uses `TfidfVectorizer` to convert phrases to numerical features.
4. **Clustering**: Applies K-Means clustering with user-defined `k` value.
5. **Output**: Displays representative phrases from each cluster for inspection.

## 🧪 Sample Output
```

Cluster 0:
fairy garden
vegetable garden
herb garden
flower garden
little garden

Cluster 1:
outdoor use
craft supplies
craft item
craft project
craft work

```

## 🛠 Technologies Used
- Python 3.x
- Pandas
- Scikit-learn
- Matplotlib (optional, for visualization)

## 🚀 How to Run
1. Clone or download this repository.
2. Ensure you have the necessary dependencies:
```

pip install -r requirements.txt

```
3. Open the project in **PyCharm** or another IDE.
4. Run `main.py` to execute the clustering process.

## 🧪 Optional Enhancements
- Visualize clusters using PCA.
- Export cluster results to CSV.
- Tune number of clusters using the elbow method.

## 📄 License
This project is for educational purposes and follows fair use guidelines for dataset application.

```