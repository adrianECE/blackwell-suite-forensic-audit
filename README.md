X# 🛡️ Blackwell Suite: AI-Driven Forensic Audit for RLHF Workforces
**:** Adrian Rapanut | **Domain:** AI Annotation for Human Data



**The Blackwell Suite** is a machine learning framework designed to maintain data integrity in Reinforcement Learning from Human Feedback (RLHF) pipelines. It identifies behavioral anomalies such as "speed-running" and "logic-faking" by experts, ensuring only high-quality human data is used for AI training.

## 🚀 Key Features
* **Behavioral Clustering:** Unsupervised K-Means clustering to discover hidden work archetypes.
* **Forensic Genealogy:** Hierarchical Dendrograms at both Task and Expert levels to map behavioral relationships.
* **Automated Fraud Prediction:** A Random Forest Classifier that predicts forensic flags with **83% Recall** on anomalous tasks.
* **Scalable Architecture:** Built for large-scale audit workflows at companies doing AI expert training and annotation.

## 📊 Technical Highlights
* **Unsupervised Phase:** Utilized the Elbow Method and Silhouette Analysis to validate 4 distinct behavioral clusters.
* **Supervised Phase:** Achieved a robust balance between Precision and Recall using ensemble learning.
* **Feature Engineering:** Focused on `similarity_score`, `time_seconds`, and `lexical_diversity` as primary forensic indicators.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, SciPy, Matplotlib, Seaborn
* **Visualization:** PCA, Hierarchical Linkage, Heatmaps

-----

## 🛠️ Installation & Environment Setup

This project is containerized using **Docker** and **WSL 2** to ensure consistent performance across different environments.

### **Prerequisites**

1.  **Docker Desktop** (with Containers/K8s enabled)
2.  **WSL 2** (Windows 11 recommended)
3.  **NVIDIA Container Toolkit** (if using GPU acceleration)

## 🚀 Technical Stack
* **Language:** Python 3.10
* **Analysis:** Scikit-learn (Random Forest, K-Means, Agglomerative Clustering)
* **Environment:** Docker & WSL 2
* **Visualization:** Streamlit, Matplotlib, Seaborn

## The following instructions assummes you have downloaded the code from Github repository and save in a directory called "Blackwell_Suite".   You may choose your own folder name.  Just take note of the path and replace accordingly.

### **Deployment Instructions**

#### **1. Access the WSL 2 Terminal**

Open your preferred WSL 2 distribution (e.g., Ubuntu) and navigate to your project directory:

```bash
cd "/mnt/c/Users/<yourUserName>/Documents/Blackwell_Suite/"
```

#### **2. Build the Docker Image**

Build the image using the provided Dockerfile. This will install all necessary dependencies (Streamlit, Scikit-Learn, etc.):

```bash
docker build -t authenti-annotator-capstone .
```

#### **3. Run the Container**

Execute the following command to launch the environment. This configuration maps the necessary ports for both Jupyter and Streamlit:

```bash
docker run --gpus all -it --rm \
    --shm-size=16g \
    --dns 8.8.8.8 \
    -p 8888:8888 -p 8501:8501 \
    -v "$(pwd):/workspace" \
    authenti-annotator-capstone
```

### **4. Run the Blackwell Suite App (app.py) using Steamlit

In Docker Desktop, Containers View, you'll see the new container running with image "authenti-annotator-capstone."

In the Action column, click the the Show Container actions, and choose opent in Terminal

In the terminal, run the Blackwell_Suite App:

```bash
streamlit run app.py
```

### **5. Accessing the Blackwell Suite**

Once the container is running, you can access the tools via your browser:

  * **Jupyter Lab/Notebook (Development):** [http://localhost:8888](https://www.google.com/search?q=http://localhost:8888)
  * **Streamlit App (Deployment/UI):** [http://localhost:8501](https://www.google.com/search?q=http://localhost:8501)

### **6. Streamlit in Jupyter Lab

You may also run the Blackwell_Suite App in Jupyter Lab. Add a new tab, and then open a Terminal

```bash
streamlit run app.py
```
-----

