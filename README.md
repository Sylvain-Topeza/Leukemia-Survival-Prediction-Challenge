# Leukemia Survival Prediction Challenge (QRT)

## 📌 Overview  
This project implements a **survival analysis model** for leukemia patients using **clinical and genomic data**.  
The goal is to predict **overall survival (OS)** from diagnosis, leveraging **machine learning & deep learning techniques**.  
This challenge is part of the **Qube Research & Technologies (QRT) Data Challenge**.

---

## ⚙️ Features  
### 1️⃣ **Data Preprocessing & Feature Engineering**  
- **Clinical Data Processing**: Standardization & imputation of missing values.  
- **Genomic Data Processing**: Mutation aggregation & feature extraction.  
- **Text-Based Embeddings**: **BioBERT embeddings** for cytogenetics interpretation.  
- **Dimensionality Reduction**: PCA applied to genomic features.  

### 2️⃣ **Survival Analysis Models**  
- **Coxnet Survival Model**: LASSO-regularized Cox model for feature selection.  
- **Random Survival Forest (RSF)**: Captures non-linear interactions between variables.  
- **Gradient Boosting Survival Analysis (GBSA)**: Boosting-based survival modeling.  
- **DeepSurv Neural Network**: A deep learning model optimized for survival data.  
- **Stacked Meta-Model**: Combines multiple models for enhanced predictions.  

### 3️⃣ **Performance Metrics & Evaluation**  
- **Concordance Index (C-Index)**: Evaluates model discrimination ability.  
- **Integrated Brier Score (IBS)**: Measures calibration & accuracy of survival predictions.  
- **Kaplan-Meier Curves**: Visualizes survival probability distributions.  

### 4️⃣ **Submission File Generation**  
- **Final predictions saved as**: `submission_stacked_meta_advanced.csv`  
- **Format**: Patient IDs & predicted survival risk scores for test data.  

---

## 📊 Methodology  

### **Step 1: Data Collection & Cleaning**  
The dataset consists of:  
✅ **Clinical features** (blood tests, demographics, cytogenetics).  
✅ **Genomic features** (mutation types, variant allele frequencies).  
✅ **Survival outcome** (OS_YEARS, OS_STATUS).  

### **Step 2: Feature Engineering**  
- **Standardization**: Normalizing numerical features for better model performance.  
- **Missing Data Handling**: Imputation techniques for missing values.  
- **BioBERT for Text Processing**: Generating embeddings from cytogenetics reports.  
- **Dimensionality Reduction**: PCA for reducing high-dimensional genomic data.  

### **Step 3: Model Training & Validation**  
Each survival model is trained and evaluated using:  
✅ **5-Fold Cross-Validation** for robust performance estimation.  
✅ **Hyperparameter Tuning** via GridSearch & Bayesian Optimization.  
✅ **Model Stacking** to improve predictive accuracy.  

### **Step 4: Final Prediction & Submission**  
- The best model (Stacked Meta-Model) generates survival risk predictions.  
- Predictions are formatted into `submission_stacked_meta_advanced.csv`.  

---

## **🔍 Example Results**  

### **1️⃣ Data Distribution & Feature Importance**  
![Feature Importance](results/feature_importance.png)  

This shows the top **clinical & genomic** predictors of survival.

### **2️⃣ Kaplan-Meier Survival Curves**  
![KM Curves](results/km_curves.png)  

Visualizing survival probability across risk groups.

### **3️⃣ Concordance Index (C-Index) Comparison**  
![C-Index](results/c_index.png)  

Model | C-Index  
------|---------  
Coxnet | 0.72  
Random Survival Forest | 0.74  
Gradient Boosting | 0.76  
DeepSurv | 0.78  
Stacked Model | **0.81**  

✅ The **Stacked Meta-Model** achieves the **highest C-Index**, outperforming individual models.  

---

## 📂 Data Source & Customization  
The dataset is **not publicly available** due to privacy regulations.  
However, users can modify the pipeline to apply it to **other survival datasets**.

---

## 🖥️ Installation & Setup  

1️⃣ **Clone the repository**  
```bash
  git clone https://github.com/yourusername/Leukemia-Survival-Prediction-Challenge.git
  cd Leukemia-Survival-Prediction-Challenge
```

2️⃣ **Install dependencies**
Requires Python 3.8+ and the following libraries:
```bash
  pip install numpy pandas matplotlib scikit-learn lifelines xgboost torch transformers
```

3️⃣ **Run the project**  
```bash
  python main.py
```
