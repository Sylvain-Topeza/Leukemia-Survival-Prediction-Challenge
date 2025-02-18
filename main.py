import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
import torch
from transformers import AutoTokenizer, AutoModel
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, backend as K

# --- Reading data ---
clinical_train = pd.read_csv('clinical_X_train.csv')
molecular_train = pd.read_csv('molecular_X_train.csv')
y_train = pd.read_csv('y_train.csv')
clinical_test = pd.read_csv('clinical_X_test.csv')
molecular_test = pd.read_csv('molecular_X_test.csv')

# --- Aggregating molecular data ---
mol_train_agg = molecular_train.groupby('ID').agg(
    mutation_count=('ID', 'count'),
    mutation_mean_vaf=('VAF', 'mean'),
    mutation_max_vaf=('VAF', 'max'),
    mutation_std_vaf=('VAF', 'std')
).reset_index()
mol_test_agg = molecular_test.groupby('ID').agg(
    mutation_count=('ID', 'count'),
    mutation_mean_vaf=('VAF', 'mean'),
    mutation_max_vaf=('VAF', 'max'),
    mutation_std_vaf=('VAF', 'std')
).reset_index()

for col in ['mutation_mean_vaf', 'mutation_max_vaf', 'mutation_std_vaf']:
    mol_train_agg[col] = mol_train_agg[col].fillna(0)
    mol_test_agg[col] = mol_test_agg[col].fillna(0)

# --- Merging clinical and molecular data ---
data_train = pd.merge(clinical_train, mol_train_agg, on='ID', how='left')
data_test = pd.merge(clinical_test, mol_test_agg, on='ID', how='left')
cols_mol = ['mutation_count', 'mutation_mean_vaf', 'mutation_max_vaf', 'mutation_std_vaf']
data_train[cols_mol] = data_train[cols_mol].fillna(0)
data_test[cols_mol] = data_test[cols_mol].fillna(0)

# --- Feature Engineering on clinical data ---
data_train['ratio_ANC_WBC'] = data_train.apply(lambda row: row['ANC'] / row['WBC'] if row['WBC'] > 0 else 0, axis=1)
data_test['ratio_ANC_WBC'] = data_test.apply(lambda row: row['ANC'] / row['WBC'] if row['WBC'] > 0 else 0, axis=1)
data_train['ANC_WBC_interaction'] = data_train['ANC'] * data_train['WBC']
data_test['ANC_WBC_interaction'] = data_test['ANC'] * data_test['WBC']

# Simple indicators from karyotype
data_train['cyto_monosomy7'] = data_train['CYTOGENETICS'].apply(lambda x: 1 if isinstance(x, str) and '7' in x else 0)
data_test['cyto_monosomy7'] = data_test['CYTOGENETICS'].apply(lambda x: 1 if isinstance(x, str) and '7' in x else 0)
data_train['cyto_complex'] = data_train['CYTOGENETICS'].apply(lambda x: 1 if isinstance(x, str) and 'complex' in x.lower() else 0)
data_test['cyto_complex'] = data_test['CYTOGENETICS'].apply(lambda x: 1 if isinstance(x, str) and 'complex' in x.lower() else 0)

# --- Extracting BioBERT embeddings for CYTOGENETICS ---
# Using the pre-trained BioBERT model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model_biobert = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model_biobert.eval()

def get_biobert_embedding(text):
    """
    Get BioBERT embedding for a given text.

    Args:
        text (str): The input text to be embedded.

    Returns:
        np.ndarray: The embedding vector.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model_biobert(**inputs)
    # Extracting the embedding of the [CLS] token
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return embedding

# Calculate embeddings for the CYTOGENETICS column
data_train['CYTOGENETICS_clean'] = data_train['CYTOGENETICS'].fillna("")
data_test['CYTOGENETICS_clean'] = data_test['CYTOGENETICS'].fillna("")

# To save time, process in batches or use a loop (this can be time-consuming)
embeddings_train = np.array([get_biobert_embedding(text) for text in data_train['CYTOGENETICS_clean']])
embeddings_test = np.array([get_biobert_embedding(text) for text in data_test['CYTOGENETICS_clean']])

# Dimensionality reduction of embeddings (e.g., to 10 dimensions)
pca_emb = PCA(n_components=10, random_state=42)
emb_train_reduced = pca_emb.fit_transform(embeddings_train)
emb_test_reduced = pca_emb.transform(embeddings_test)

# Convert to DataFrame and concatenate
emb_train_df = pd.DataFrame(emb_train_reduced, columns=[f'biobert_pca_{i}' for i in range(10)], index=data_train.index)
emb_test_df = pd.DataFrame(emb_test_reduced, columns=[f'biobert_pca_{i}' for i in range(10)], index=data_test.index)
data_train = pd.concat([data_train, emb_train_df], axis=1)
data_test = pd.concat([data_test, emb_test_df], axis=1)

# --- Merge with training targets ---
data_train = pd.merge(data_train, y_train, on='ID', how='left')
data_train = data_train.dropna(subset=['OS_STATUS', 'OS_YEARS'])

# --- Selection of final features ---
features = [
    'BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT',
    'mutation_count', 'mutation_mean_vaf', 'mutation_max_vaf', 'mutation_std_vaf',
    'ratio_ANC_WBC', 'ANC_WBC_interaction', 'cyto_monosomy7', 'cyto_complex'
] + [f'biobert_pca_{i}' for i in range(10)]

X_train = data_train[features]
X_test = data_test[features]

# --- Imputation and Standardization ---
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

# Preparing the target for survival analysis (structured format)
y_train_struct = np.array(
    [(bool(event), time) for event, time in zip(data_train['OS_STATUS'], data_train['OS_YEARS'])],
    dtype=[('event', 'bool'), ('time', 'float')]
)

print("Step 1 completed: Preprocessing and enriched features done.")

# 1. CoxnetSurvivalAnalysis
coxnet = CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=np.logspace(-4, 0, 100))
coxnet.fit(X_train_scaled, y_train_struct)
coxnet_risk_train = coxnet.predict(X_train_scaled)
coxnet_risk_test = coxnet.predict(X_test_scaled)
print("Coxnet model trained.")

# 2. Random Survival Forest (RSF)
rsf = RandomSurvivalForest(
    n_estimators=250,
    max_features='sqrt',
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
rsf.fit(X_train_scaled, y_train_struct)
rsf_risk_train = rsf.predict(X_train_scaled)
rsf_risk_test = rsf.predict(X_test_scaled)
print("RSF model trained.")

# 3. Gradient Boosting Survival Analysis (GBSA)
gbsa = GradientBoostingSurvivalAnalysis(
    n_estimators=300,
    learning_rate=0.01,
    max_depth=4,
    random_state=42
)
gbsa.fit(X_train_scaled, y_train_struct)
gbsa_risk_train = gbsa.predict(X_train_scaled)
gbsa_risk_test = gbsa.predict(X_test_scaled)
print("GBSA model trained.")

# 4. Improved DeepSurv with Keras
def neg_log_partial_likelihood(y_true, y_pred):
    """
    Custom negative log partial likelihood loss function for survival analysis.

    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted values.

    Returns:
        tf.Tensor: Computed loss.
    """
    events = y_true[:, 0]
    exp_y = K.exp(y_pred)
    cum_sum = K.cumsum(exp_y[::-1])[::-1]
    log_risk = K.log(cum_sum + K.epsilon())
    loss = -K.sum((y_pred - log_risk) * events) / (K.sum(events) + K.epsilon())
    return loss

def build_deepsurv_model(input_shape):
    """
    Build the DeepSurv model.

    Args:
        input_shape (int): Shape of the input features.

    Returns:
        tf.keras.Model: Compiled DeepSurv model.
    """
    inputs = layers.Input(shape=(input_shape,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='linear')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss=neg_log_partial_likelihood)
    return model

# Preparing labels for DeepSurv
y_train_ds = data_train[['OS_STATUS', 'OS_YEARS']].to_numpy().astype('float32')
# Sort by descending survival time
sort_idx = np.argsort(-y_train_ds[:, 1])
X_train_ds = X_train_scaled[sort_idx]
y_train_ds = y_train_ds[sort_idx]

input_dim = X_train_ds.shape[1]
deepsurv_model = build_deepsurv_model(input_dim)
history = deepsurv_model.fit(X_train_ds, y_train_ds, epochs=100, batch_size=64, validation_split=0.2, verbose=1)

deep_risk_train = deepsurv_model.predict(X_train_scaled).flatten()
deep_risk_test = deepsurv_model.predict(X_test_scaled).flatten()
print("DeepSurv model trained.")

print("Step 2 completed: Survival models trained.")

# Modified custom loss function for the meta-model
def neg_log_partial_likelihood(y_true, y_pred):
    events = y_true[:, 0]
    # Clip y_pred to avoid overflow in exponentiation
    y_pred_clipped = K.clip(y_pred, -20, 20)
    exp_y = K.exp(y_pred_clipped)
    cum_sum = K.cumsum(exp_y[::-1])[::-1]
    log_risk = K.log(cum_sum + K.epsilon())
    loss = -K.sum((y_pred_clipped - log_risk) * events) / (K.sum(events) + K.epsilon())
    return loss

# Preparing stacked prediction matrices from all base models
stacked_train = np.column_stack((coxnet_risk_train, rsf_risk_train, gbsa_risk_train, deep_risk_train))
stacked_test = np.column_stack((coxnet_risk_test, rsf_risk_test, gbsa_risk_test, deep_risk_test))

# Building the non-linear meta-model with Keras
def build_meta_model(input_shape):
    """
    Build the meta-model for stacking.

    Args:
        input_shape (int): Shape of the input features.

    Returns:
        tf.keras.Model: Compiled meta-model.
    """
    inputs = layers.Input(shape=(input_shape,))
    x = layers.Dense(16, activation='relu')(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(8, activation='relu')(x)
    outputs = layers.Dense(1, activation='linear')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss=neg_log_partial_likelihood)
    return model

meta_model = build_meta_model(stacked_train.shape[1])
# Training the meta-model with internal validation
meta_model.fit(stacked_train, y_train_ds, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Final prediction via the meta-model
final_risk_scores = meta_model.predict(stacked_test).flatten()

# Creating the submission file
submission = pd.DataFrame({'ID': data_test['ID'], 'risk_score': final_risk_scores})
submission.set_index('ID', inplace=True)
submission.to_csv('challenge_submission.csv')

print("Step 3 completed: challenge_submission.csv file generated.")
