# 📊 Customer Churn Analysis and Segmentation Using Machine Learning

### An End-to-End Machine Learning Project for Customer Behavior Analysis, Churn Prediction and Customer Segmentation.

<p align="center">
  <a href=https://customer-churn-analysis-prediction-and-segmentation-hhlcndb3sz.streamlit.app/
    🚀 <strong>Live Application – Try It Here</strong>
  </a>
</p>

---

## 1. 📌 Project Overview

Customer churn is one of the major challenges faced by businesses, particularly in subscription-based industries. It refers to the situation where customers stop using a company's products or services.

Retaining existing customers is important for maintaining business growth, improving customer satisfaction, and reducing customer acquisition costs.

This project, **Customer Churn Analysis and Segmentation Using Machine Learning**, focuses on analyzing customer behavior, identifying churn patterns, predicting potential customer churn, and grouping customers into different segments using machine learning techniques.

The project uses supervised and unsupervised machine learning algorithms to provide insights into customer behavior and support data-driven business decisions.

The application is developed using Python and Streamlit, allowing users to explore customer data, predict churn for individual customers, and visualize customer segments through an interactive dashboard.

**Live Application:**

https://customer-churn-analysis-prediction-and-segmentation-hhlcndb3sz.streamlit.app/

---

## 2. 🎯 Business Problem

Businesses often face difficulties identifying customers who are likely to discontinue their services.

Traditional approaches to customer retention may not identify potential churners early enough because customer behavior involves multiple factors, such as:

- Customer age
- Gender
- Customer tenure
- Usage frequency
- Number of support calls
- Subscription type
- Contract length
- Payment delays
- Total customer spending
- Last interaction

Without proper analysis, businesses may struggle to understand customer behavior and identify patterns associated with customer churn.

Therefore, this project aims to use machine learning techniques to analyze customer information, predict potential churn, and identify meaningful customer segments.

---

## 3. 🎯 Project Objectives

The major objectives of this project are:

1. Perform Exploratory Data Analysis (EDA) to understand customer behavior and identify important patterns.
2. Clean and preprocess the dataset by handling missing values, categorical variables, and numerical features.
3. Develop machine learning classification models to predict customer churn.
4. Evaluate model performance using appropriate classification metrics.
5. Apply clustering techniques to identify different customer segments.
6. Visualize customer clusters using dimensionality reduction techniques.
7. Develop an interactive Streamlit application for customer churn prediction and segmentation.
8. Containerize the application using Docker.

---

## 4. 📂 Dataset Description

The project uses a customer churn dataset containing approximately **80,000 customer records**.

The dataset includes customer-related information that helps analyze customer behavior and predict churn.

### Dataset Features

| Feature | Description |
|---|---|
| CustomerID | Unique customer identifier |
| Age | Age of the customer |
| Gender | Gender of the customer |
| Tenure | Duration of the customer's relationship with the company |
| Usage Frequency | Frequency of customer service usage |
| Support Calls | Number of support calls made by the customer |
| Subscription Type | Type of subscription selected |
| Contract Length | Duration of the customer's contract |
| Payment Delay | Number of days of payment delay |
| Total Spend | Total amount spent by the customer |
| Last Interaction | Number of days since the last customer interaction |
| Churn | Target variable indicating whether the customer has churned |

### Target Variable

- `0` – Customer Retained
- `1` – Customer Churned

---

## 5. 🛠️ Technology Stack

The following technologies and libraries are used in this project.

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning algorithms and preprocessing |
| Streamlit | Interactive web application |
| Joblib | Saving and loading trained models |
| Git | Version control |
| GitHub | Source code management |
| Docker | Application containerization |

---

## 6. 🤖 Machine Learning Algorithms

This project uses both supervised and unsupervised machine learning techniques.

### 6.1 Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to generate predictions.

In this project, Random Forest is used as the primary classification model for predicting customer churn.

**Key Responsibilities:**

- Learn patterns from historical customer data.
- Classify customers based on their likelihood of churning.
- Generate churn predictions.
- Estimate churn probabilities.

### 6.2 Decision Tree Classifier

Decision Tree is a supervised machine learning algorithm that makes predictions using a tree-like structure of decision rules.

In this project, Decision Tree is implemented as an additional classification model.

It helps provide another model for comparison and understanding classification behavior.

### 6.3 DBSCAN Clustering

DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise.

It is an unsupervised machine learning algorithm that groups data points based on their density.

In this project, DBSCAN is used for customer segmentation.

**Key Responsibilities:**

- Identify customer groups based on similarities in their features.
- Discover density-based clusters.
- Identify customers that do not belong to any cluster.
- Support customer behavior analysis.

> **Important:** DBSCAN assigns `-1` to noise points. This represents observations that do not belong to an identified cluster; it does not indicate customer churn.

### 6.4 Principal Component Analysis (PCA)

PCA is a dimensionality reduction technique used to transform high-dimensional data into a smaller number of components.

In this project, PCA is used to reduce customer features into two dimensions for visualizing DBSCAN clusters.

---

## 7. ⚙️ Project Workflow

The overall project workflow follows these stages:

### Step 1: Data Collection

Load the customer churn dataset containing approximately 80,000 records.

### Step 2: Exploratory Data Analysis (EDA)

Analyze the dataset to understand:

- Dataset dimensions.
- Data types.
- Missing values.
- Duplicate records.
- Numerical feature distributions.
- Categorical feature distributions.
- Relationships between customer attributes and churn.

### Step 3: Data Preprocessing

Prepare the dataset for machine learning by:

- Handling missing values.
- Handling duplicate customer identifiers.
- Encoding categorical variables.
- Scaling numerical features.
- Separating independent and dependent variables.

### Step 4: Model Development

Develop classification models using Random Forest and Decision Tree algorithms.

### Step 5: Model Evaluation

Evaluate classification performance using suitable evaluation metrics.

### Step 6: Customer Segmentation

Apply DBSCAN clustering to identify customer groups based on their behavioral characteristics.

### Step 7: Cluster Visualization

Use PCA to visualize customer segments in a two-dimensional representation.

### Step 8: Application Development

Develop an interactive Streamlit application integrating the trained models and visualizations.

### Step 9: Deployment and Containerization

Deploy the application using Streamlit Community Cloud and containerize it using Docker.

---

## 8. 📁 Project Structure

The repository follows an organized folder structure to manage the application, dataset, trained models, documentation and source files.

```text
Churn Analysis/
│
├── data/
│   └── customer_churn_dataset_80000_records.csv
│
├── Docs/
│   ├── decision_tree.png
│   ├── RandomForest_decision_tree_1.png
│   └── RandomForest_decision_tree_200.png
│
├── models/
│   ├── dbscan_model.pkl
│   ├── DT_model.pkl
│   ├── MinMax_scaler.pkl
│   ├── rf_model (1).pkl
│   └── scaler.pkl
│
├── Notebooks/
│   └── Customer_Churn_Analysis_Segmentation.ipynb
│
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

### Folder Description

| Folder/File | Description |
|---|---|
| `app/` | Contains the main Streamlit application. |
| `data/` | Stores the customer churn dataset. |
| `Docs/` | Contains business objectives, project images that developed by using Decision Tree and Random Forest Algorithm |
| `models/` | Stores trained machine learning models and preprocessing scalers. |
| `Notebooks/` | Contains the Jupyter Notebook used for data analysis and model development. |
| `.gitignore` | Specifies files and folders that should not be tracked by Git. |
| `Dockerfile` | Contains instructions for building the Docker image. |
| `requirements.txt` | Contains the required Python libraries and dependencies. |
| `README.md` | Provides project information and setup instructions. |

---

## 9. 🖥️ Streamlit Application Features

The project provides an interactive web application with the following sections.

### 9.1 Dashboard

The dashboard provides an overview of customer information and churn patterns.

**It includes:**

- Total customer count.
- Churned customer count.
- Retained customer count.
- Churn distribution.
- Customer-related visualizations.
- Subscription and contract analysis.

### 9.1 Dashboard

<img width="1920" height="1008" alt="Screenshot 2026-11-13 224148" src="https://github.com/user-attachments/assets/a085e324-a9e8-467f-8fb1-08f1039a3c0c" />

### 9.2 Churn Prediction

This section allows users to enter individual customer details.

The trained Random Forest model processes the customer information and generates:

- Churn prediction.
- Churn probability.
- Customer retention prediction.

This feature demonstrates how a trained machine learning model can be used for individual customer predictions.

<img width="1920" height="1008" alt="Screenshot 2026-11-13 224148" src="https://github.com/user-attachments/assets/5fa16d4e-76d8-450d-8a41-87a8a1d38cc4" />

### 9.3 Customer Segmentation

This section focuses on identifying customer groups using DBSCAN.

**It provides:**

- Cluster distribution.
- Number of identified clusters.
- Noise-point information.
- Customer segment analysis.
- PCA-based cluster visualization.

<img width="1920" height="1080" alt="Screenshot 2026-11-13 224208" src="https://github.com/user-attachments/assets/69856573-f9b1-4a47-a58f-b6b59e836ed7" />

### 9.4 Model Information

This section provides information about the machine learning algorithms used in the project.

**It includes details about:**

- Random Forest Classifier.
- Decision Tree Classifier.
- DBSCAN Clustering.

<img width="1920" height="1080" alt="Screenshot 2026-11-13 224227" src="https://github.com/user-attachments/assets/5686f6bb-794b-4141-833c-6428ebc95ae3" />

---
These images are just for understand and there are more images are present you can see when you run the code.

## 10. 📈 Model Evaluation Metrics

Different evaluation metrics are used to assess the performance of supervised and unsupervised machine learning algorithms.

### 10.1 Classification Evaluation Metrics

The Random Forest and Decision Tree classification models can be evaluated using the following metrics:

| Metric | Description |
|---|---|
| Accuracy | Measures the proportion of correctly predicted observations. |
| Precision | Measures how many predicted positive cases are actually positive. |
| Recall | Measures how many actual positive cases are correctly identified. |
| F1-Score | Harmonic mean of precision and recall. |
| Confusion Matrix | Visualizes correct and incorrect classification results. |
| ROC-AUC Score | Measures the model's ability to distinguish between churned and retained customers across different classification thresholds. |

### 10.2 Clustering Evaluation Metrics

DBSCAN clustering performance can be evaluated using:

| Metric | Description |
|---|---|
| Silhouette Score | Measures how well customer observations fit within their assigned clusters compared with other clusters. |
| Number of Clusters | Represents the number of customer groups identified by DBSCAN. |
| Noise Points | Represents observations that do not belong to any identified cluster. |

### Evaluation Purpose

- **ROC-AUC Score:** Evaluates the classification model's ability to distinguish between churned and retained customers.
- **Silhouette Score:** Evaluates the quality and separation of customer clusters generated by DBSCAN.
- **Number of Clusters:** Helps understand how many customer groups DBSCAN has identified.
- **Noise Points:** Helps identify observations that DBSCAN could not assign to a cluster.

> **Note:** ROC-AUC is applicable to classification models, while Silhouette Score is applicable to clustering models. For DBSCAN, noise points (label `-1`) should generally be excluded when calculating the Silhouette Score. A Silhouette Score also requires at least two valid clusters.

---

## 11. 🚀 Installation and Execution

Follow these steps to run the project locally.

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### Step 2: Navigate to the Project Directory

```bash
cd Customer-Churn-Analysis-and-Segmentation
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 5: Run the Streamlit Application

```bash
streamlit run app/app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 12. 🐳 Docker Deployment

Docker is used to containerize the application along with its required dependencies.

### Step 1: Build the Docker Image

Run the following command from the main project directory:

```bash
docker build -t customer-churn-analysis .
```

### Step 2: Run the Docker Container

```bash
docker run -p 8201:8201 customer-churn-analysis
```

### Step 3: Access the Application

Open the following address in your browser:

```text
http://localhost:8201
```

---

## 13. ☁️ Streamlit Cloud Deployment

The application is deployed using Streamlit Community Cloud, making it accessible through a web browser.

### Deployment Configuration

| Configuration | Value |
|---|---|
| Platform | Streamlit Community Cloud |
| Repository | Customer-Churn-Analysis-and-Segmentation |
| Branch | main |
| Main File Path | app/app.py |

### Live Application

🚀 [Click Here to Access the Customer Churn Prediction Application](https://customer-churn-analysis-prediction-and-segmentation-hhlcndb3sz.streamlit.app/)

---

## 14. 📈 Business Impact

This project demonstrates how machine learning can help businesses improve customer retention strategies.

### Key Benefits

- **Early Churn Identification:** Identify customers predicted to discontinue their services.
- **Customer Behavior Analysis:** Understand customer usage patterns and churn-related factors.
- **Customer Segmentation:** Discover groups of customers with similar characteristics.
- **Data-Driven Decisions:** Support business decisions using machine learning predictions and analytical insights.
- **Retention Strategies:** Help businesses identify customers who may need additional attention.

**Ultimate Goal:** Move from reactive customer management to proactive, data-driven customer retention.

---

## 15. Developed By:

**Lokesh Vallapu**
