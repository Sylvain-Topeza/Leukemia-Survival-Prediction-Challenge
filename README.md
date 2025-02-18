# 🔬 Overall Survival Prediction for Patients with Myeloid Leukemia

## 📌 Challenge Overview

This project is part of a data science challenge aimed at predicting **overall survival (OS) time** for patients diagnosed with **myeloid leukemia** based on **clinical and molecular data**. 

The goal is to develop models that estimate survival probabilities, which can help clinicians personalize treatment plans for patients. Accurate predictions of overall survival can assist in:

- Identifying **low-risk patients** who may receive less intensive therapies.
- Identifying **high-risk patients** who may require aggressive treatments such as **hematopoietic stem cell transplantation**.

The dataset consists of:
- **Clinical Data**: Blood biomarkers, hematological values, and patient demographics.
- **Molecular Data**: Genetic mutations specific to cancerous cells.
- **Survival Data**: OS time in years and survival status (event/censoring).

🔗 [Challenge page link](https://challengedata.ens.fr/challenges/162)

---

## 🚀 Features and Methodology

### **1. Data Processing**
- **Merging Clinical and Molecular Data**: The dataset is structured such that clinical and molecular data are merged using a unique `ID` per patient.
- **Mutation Aggregation**: Summary statistics such as **mutation count, mean Variant Allele Frequency (VAF), max VAF, and VAF standard deviation** are computed for each patient.
- **Feature Engineering**: Additional features are created, including:
  - **Ratio of Absolute Neutrophil Count (ANC) to White Blood Cell Count (WBC)**.
  - **ANC * WBC interaction term**.
  - **Cytogenetic Markers** such as **monosomy 7 presence** and **complex karyotype identification**.

### **2. Text Embedding Extraction for Cytogenetics**
- **Using BioBERT**: The `CYTOGENETICS` column (textual cytogenetic descriptions) is converted into numerical embeddings using **BioBERT**.
- **Dimensionality Reduction (PCA)**: The BioBERT embeddings are compressed into **10 principal components** for model compatibility.

### **3. Survival Modeling**
Several models are used for survival prediction:

- **Cox Proportional Hazards Model (Coxnet)**: Regularized Cox model with **elastic net penalty** to manage feature selection and multicollinearity.
  - `l1_ratio=0.5` (compromise between L1 and L2 regularization).
  - `alphas=np.logspace(-4, 0, 100)` (range of penalization strengths).

- **Random Survival Forest (RSF)**: Ensemble-based model capturing **non-linear interactions** between covariates.
  - `n_estimators=250` (number of trees for stability).
  - `max_features='sqrt'` (reduces overfitting by limiting available features per split).

- **Gradient Boosting Survival Analysis (GBSA)**: Gradient boosting approach for survival tasks.
  - `n_estimators=300`, `learning_rate=0.01` (prevents overfitting while converging smoothly).
  - `max_depth=4` (moderate tree depth to prevent overfitting).

- **DeepSurv (Neural Network for Survival Analysis)**: Deep learning model learning complex survival patterns.
  - Three **Dense layers (128, 64, 32 neurons)** with **Dropout (40%)**.
  - Optimized using **Adam optimizer** and a **custom survival loss function**.

### **4. Model Stacking & Meta-Modeling**
- **Stacking Approach**: Risk scores from all models are stacked to form a meta-feature matrix.
- **Meta-Model (Neural Network)**: A fully connected neural network (2 hidden layers, **Dropout 30%**) is trained on the stacked predictions to optimize the final survival prediction.

---

## 🛠 Installation & Execution

### **Prerequisites**
This project requires **Python 3.8+** and the following dependencies:
```bash
pip install pandas numpy scikit-learn transformers torch tensorflow sksurv
```

### **Run the Code**
1. Download the dataset files(`clinical_X_train.csv`, `molecular_X_train.csv`, `y_train.csv`, `clinical_X_test.csv`, `molecular_X_test.csv`) from the [challenge website](https://challengedata.ens.fr/challenges/162) and place them in the project directory
2. Execute the main script:
```bash
python main.py
```
3. The final submission file **`challenge_submission.csv`** will be generated.

---

## 📌 Potential Improvements
- **Hyperparameter Optimization** (e.g., GridSearch for RSF, GBSA)
- **Data Augmentation** for clinical data (synthetic generation, missing data imputation)
- **Additional NLP models** (SciBERT for better text feature extraction)
