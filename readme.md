
# 📊 Customer Churn Analysis and Segmentation Using Machine Learning

### An End-to-End Machine Learning Project for Customer Churn Prediction and Customer Segmentation

<p align="center">
  <a href="https://customer-churn-analysis-prediction-and-segmentation.streamlit.app/">
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
https://customer-churn-analysis-prediction-and-segmentation.streamlit.app/

---

## 2. 🎯 Business Problem

Businesses often face difficulties identifying customers who are likely to discontinue their services in that organization.

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

Develop classification models using Random Forest and Decision Tree Algorithms.

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
├── Src/
│   ├── __init__.py
│   ├── prediction.py
│   ├── preprocessing.py
│   └── segmentation.py
│
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt

### Folder Description

| Folder/File | Description |
|---|---|
| `app/` | Contains the main Streamlit application. |
| `data/` | Stores the customer churn dataset. |
| `Docs/` | Contains the images that are developed by using the Decision tree and Random Forest Algorithms |
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

![Dashboard]
<img width="1920" height="1080" alt="Screenshot 2026-11-13 224102" src="https://github.com/user-attachments/assets/116e09ba-95db-4f52-a555-44320f3605b7" />

### 9.2 Churn Prediction

This section allows users to enter individual customer details.

The trained Random Forest model processes the customer information and generates:

- Churn prediction.
- Churn probability.
- Customer retention prediction.

This feature demonstrates how a trained machine learning model can be used for individual customer predictions.

![Churn Prediction]
<img width="1920" height="1008" alt="Screenshot 2026-11-13 224148" src="https://github.com/user-attachments/assets/05d30921-cfa1-43eb-aeec-45b83d0f1f52" />

### 9.3 Customer Segmentation

This section focuses on identifying customer groups using DBSCAN.

**It provides:**

- Cluster distribution.
- Number of identified clusters.
- Noise-point information.
- Customer segment analysis.
- PCA-based cluster visualization.

![Customer Segmentation]
<img width="1920" height="1080" alt="Screenshot 2026-11-13 224208" src="https://github.com/user-attachments/assets/542babdc-f976-40ae-bd76-5169a01c3c47" />

### 9.4 Model Information

This section provides information about the machine learning algorithms used in the project.

**It includes details about:**

- Random Forest Classifier.
- Decision Tree Classifier.
- DBSCAN Clustering.

![Model Information]
<img width="1920" height="1080" alt="Screenshot 2026-11-13 224227" src="https://github.com/user-attachments/assets/9ff72414-05b1-4708-a512-34dfe5503eaa" />
<img width="1920" height="1080" alt="Screenshot 2026-11-13 224239" src="https://github.com/user-attachments/assets/d6797ec2-6043-4122-ad35-422e33f0caa5" />

---

## 10. 📈 Model Evaluation Metrics

Classification models can be evaluated using the following metrics:

| Metric | Description |
|---|---|
| Accuracy | Measures the proportion of correct predictions. |
| Precision | Measures how many predicted positive cases are actually positive. |
| Recall | Measures how many actual positive cases are correctly identified. |
| F1-Score | Harmonic mean of precision and recall. |
| Confusion Matrix | Shows correct and incorrect classification results. |

These metrics help evaluate model performance from different perspectives.
