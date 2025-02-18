# Leukemia Survival Prediction Challenge (QRT)

This repository contains the implementation of **Leukemia Survival Prediction Challenge**, a data science competition organized by **Qube Research & Technologies (QRT)**. The objective is to develop **machine learning models** to predict **overall survival (OS)** of patients diagnosed with leukemia using **clinical and molecular** data.

## 🚀 Project Overview

The goal of this challenge is to build robust survival analysis models using **clinical features, genomic embeddings (BioBERT), and machine learning techniques**. The dataset consists of:
- **Clinical Features**: Blood test values, cytogenetics, and demographics.
- **Genomic Data**: Mutations and variant allele frequencies.
- **Survival Data**:
  - `OS_YEARS`: The duration from diagnosis to last follow-up or death.
  - `OS_STATUS`: The event indicator (1 = deceased, 0 = censored).

## 📊 Data Preprocessing & Feature Engineering

The preprocessing pipeline includes:
✅ **Mutation Aggregation**: Aggregate molecular data per patient (mutation count, mean, max, std).  
✅ **Feature Engineering**: Creation of clinical interaction terms (`ANC/WBC ratio`, `cytogenetics flags`).  
✅ **Genomic Embeddings**: Using **BioBERT** to generate **text embeddings** from cytogenetics reports.  
✅ **Dimensionality Reduction**: Applying **PCA** to BioBERT embeddings (10 principal components).  
✅ **Standardization & Imputation**: Using `SimpleImputer` and `StandardScaler`.

## 🤖 Models Used

Several survival analysis models were trained:

| Model                           | Description |
|---------------------------------|-------------|
| **Coxnet Survival Analysis**    | Lasso-penalized Cox regression for feature selection. |
| **Random Survival Forest (RSF)** | Ensemble learning model capturing interactions between variables. |
| **Gradient Boosting Survival Analysis (GBSA)** | Boosting-based approach for time-to-event modeling. |
| **DeepSurv (Neural Network)**   | Deep learning model optimized for survival analysis. |
| **Stacked Meta-Model**          | A final model aggregating all base models for improved predictions. |

## 🛠 Installation & Usage

### 📌 Prerequisites
- Python 3.8+
- Libraries listed in `requirements.txt`

### 🏗️ Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Leukemia-Survival-Prediction-Challenge.git
cd Leukemia-Survival-Prediction-Challenge

# Install dependencies
pip install -r requirements.txt

