# AI in Healthcare (CSET343) — Lab Assignments

**Course:** B.Tech Specialization Elective (4th Year, Semester VII)  
**Course Code:** CSET343 — AI in Healthcare  
**Repository:** [achalzz/ai-healthcare-cset343](https://github.com/achalzz/ai-healthcare-cset343)

---

## Lab Assignment 1: Data Cleaning & Preprocessing

**Notebook:** [`Lab_1_Healthcare.ipynb`](Lab_1_Healthcare.ipynb)  
**Dataset:** Heart Failure Clinical Records

### What Was Asked

> Perform data cleaning and preprocessing on a clinical dataset — handle missing values, remove outliers, encode categorical variables, normalize features, compute summary statistics before and after cleaning, and validate with visualizations.

### What We Did

| # | Requirement | Implementation |
|---|-------------|----------------|
| 1 | Download Heart Failure dataset | Loaded via UCI ML Repository / Kaggle backup with multi-tier fallback |
| 2 | Load and display first 5 rows | Loaded as Pandas DataFrame, displayed `.head()` with clinical column descriptions |
| 3 | Summarize dataset (rows, columns, dtypes, missing) | Computed `.info()`, `.describe()`, missing value counts, data types report |
| 4 | Visualize missing values | Plotted missing value heatmap and bar chart |
| 5 | Plot numerical distributions | KDE plots, histograms, and boxplots for all numerical features |
| 6 | Impute missing values | Applied KNN Imputer (k=5) for clinically sound multivariate imputation |
| 7 | Remove outliers | IQR-based outlier detection and removal with clinical justification |
| 8 | Correct inconsistencies | Checked and fixed invalid entries (e.g., negative age, impossible lab values) |
| 9 | Feature engineering | Created `age_group` bins (<40, 40–60, >60) and `risk_score` (normalized product of ejection_fraction × serum_creatinine) |
| 10 | Encode categorical variables | Applied label encoding to `sex`, `smoking`, and other categorical features |
| 11 | Normalize features | MinMaxScaler applied to scale numerical features to [0, 1] |
| 12 | Compute summary statistics before/after | Mean, median, std comparison tables pre- and post-cleaning |
| 13 | Validate cleaning with plots | Histograms and boxplots for cleaned numerical features |
| 14 | Confirm no missing values remain | Final `.isna().sum()` verification confirming 0 remaining NaNs |

---

## Lab Assignment 2: Data Cleaning & Clinical Data Enrichment

**Notebook:** [`Lab_Assignment_2_AI_in_Healthcare.ipynb`](Lab_Assignment_2_AI_in_Healthcare.ipynb)  
**Dataset:** Heart Failure Clinical Records

### What Was Asked

> Perform data cleaning and enrichment on a sample clinical dataset with missing values — download, load, summarize, visualize missing values, plot distributions, impute missing values, remove outliers, correct inconsistencies, engineer features, encode categoricals, normalize, compute statistics before/after, validate cleaning, and confirm no missing values.

### What We Did

| # | Requirement | Implementation |
|---|-------------|----------------|
| 1–2 | Dataset Acquisition & Loading | Downloaded Heart Failure dataset with multi-tier fallback (Kaggle API → direct URL → UCI backup), loaded into Pandas DataFrame |
| 3 | Summarize Dataset | Reported rows (299), columns (13), data types, missing value counts per column |
| 4 | Visualize Missing Values | Heatmap (Seaborn) and bar chart showing missing value distribution per feature |
| 5 | Plot Numerical Distributions | KDE plots, histograms, and boxplots stratified by death event outcome |
| 6 | Impute Missing Values | KNN Imputer (k=5) with distribution verification before/after imputation |
| 7–8 | Outlier Removal & Inconsistency Correction | IQR-based outlier detection, clinical range validation (e.g., ejection fraction 10–80%, creatinine > 0) |
| 9 | Feature Engineering | Created `age_group` (bins: <40, 40–60, >60), `risk_score` (composite clinical index), `anemia_diabetes_interaction` |
| 10 | Encode Categorical Variables | One-hot and label encoding for `sex`, `smoking`, `diabetes`, `anaemia` |
| 11 | Normalize Features | MinMaxScaler applied to all continuous numerical features |
| 12 | Compute Summary Statistics | Before/after comparison tables (mean, median, std, min, max) |
| 13 | Validate Cleaning | Post-cleaning histograms and boxplots for all numerical features |
| 14 | Confirm Missing Values | Final audit confirming 0 NaNs remaining in cleaned dataset |

---

## Lab Assignment 3: Multi-Modality Data Processing & Analysis

**Notebook:** [`Lab_Assignment_3_AI_in_Healthcare.ipynb`](Lab_Assignment_3_AI_in_Healthcare.ipynb)  
**Datasets:** Pima Indians Diabetes (Tabular), MTSamples (Textual), Chest X-Ray (Image), MIT-BIH Arrhythmia (Signal/ECG)

### What Was Asked

> Perform data acquisition, cleaning, preprocessing and analysis on four different healthcare data modalities: Tabular, Textual, Image, and Signal data. Apply modality-specific cleaning, preprocessing into model-ready format, hypothesis testing (Chi-square, ANOVA), EDA, feature engineering, feature selection/extraction, noise removal, and augmentation.

### What We Did

| Part | Modality | Requirement | Implementation |
|------|----------|-------------|----------------|
| **A** | Tabular (Pima Diabetes) | Acquire, clean, preprocess, hypothesis tests, EDA | Loaded 768-patient dataset, replaced biologically impossible zeros with NaN, KNN imputed, computed Chi-square test (categorical vs Outcome) and one-way ANOVA (continuous features across diabetic groups), correlation heatmap, feature distributions stratified by diabetes status |
| **B** | Textual (MTSamples) | Acquire medical transcriptions, clean text, NLP preprocessing | Downloaded MTSamples clinical transcriptions, removed PHI placeholders, lowercased, tokenized, stopword removal, lemmatization, TF-IDF vectorization, word cloud visualization, top medical specialty frequency analysis |
| **C** | Image (Chest X-Ray) | Acquire images, clean metadata, resize, augmentation | Loaded Normal/Pneumonia chest X-ray images, resized to 128×128 grayscale, pixel intensity normalization [0,1], applied data augmentation (rotation, flip, zoom), sample image grid visualization with class labels |
| **D** | Signal (MIT-BIH ECG) | Acquire ECG signals, clean, denoise, segment | Loaded MIT-BIH Arrhythmia Database via `wfdb`, applied bandpass filter (0.5–45 Hz) for noise removal, R-peak detection, heartbeat segmentation into fixed-length windows, ECG waveform visualization with annotated beat labels |

---

## Lab Assignment 4: Binary Classification with Logistic Regression

**Notebook:** [`Lab_Assignment_4_AI_in_Healthcare.ipynb`](Lab_Assignment_4_AI_in_Healthcare.ipynb)  
**Dataset:** Breast Cancer Wisconsin (Diagnostic)

### What Was Asked

> Hands-on experience with logistic regression for binary classification — load the Breast Cancer Wisconsin dataset, compute summary statistics, handle missing/duplicates/outliers, visualize data distributions and correlations, discuss class imbalance, preprocess with stratified split and standardization, train logistic regression, evaluate with accuracy/precision/recall/F1/confusion matrix/ROC-AUC, and explore regularization and model interpretation.

### What We Did

| Task | Requirement | Implementation |
|------|-------------|----------------|
| **Task 1: Data Acquisition & Exploration** | Download dataset, summary stats, missing/duplicate/outlier check, visualizations, class imbalance discussion | Loaded 569-patient dataset (30 continuous features) via `sklearn.datasets.load_breast_cancer`, computed mean/median/std/skewness for all 30 features, verified 0 missing values and 0 duplicates, IQR outlier detection, KDE distributions stratified by Malignant/Benign, 30×30 correlation heatmap, class balance pie chart (212 Malignant / 357 Benign) |
| **Task 2: Data Preprocessing** | Binary target encoding, stratified 80/20 split, feature standardization | Mapped target labels to binary (Malignant=1, Benign=0), `train_test_split` with `stratify=y`, `StandardScaler` fit on training set only (preventing data leakage) |
| **Task 3: Model Training & Evaluation** | Train logistic regression, Accuracy, Precision, Recall, F1, Confusion Matrix, ROC Curve, AUC | Trained `LogisticRegression`, computed all metrics, plotted annotated confusion matrix with Sensitivity/Specificity/PPV/NPV labels, ROC curve with AUC score |
| **Task 4: Advanced Concepts** | Regularization (L1 vs L2), model interpretation | L1 (Lasso) vs L2 (Ridge) `GridSearchCV` comparison across regularization strengths, clinical decision threshold optimization for 100% malignant recall, feature importance via Odds Ratios ($e^\beta$), individual patient sigmoidal case studies |

---

## Lab Assignment 5: Multiclass Classification in Healthcare

**Notebook:** [`Lab_Assignment_5_AI_in_Healthcare.ipynb`](Lab_Assignment_5_AI_in_Healthcare.ipynb)  
**Dataset:** UC Irvine Dermatology Dataset (6-class skin diseases)

### What Was Asked

> Hands-on experience with multiclass classification — load Dermatology or Vertebral dataset, compute summary statistics, handle missing values, visualize distributions/correlations/class balance, preprocess with stratified split and standardization, implement 3–5 different ML algorithms (KNN, Random Forest, Logistic Regression, SVM, etc.), evaluate with Accuracy/Precision/Recall/F1/Confusion Matrix/ROC-AUC, compare and interpret results.

### What We Did

| Task | Requirement | Implementation |
|------|-------------|----------------|
| **Task 1: Data Acquisition & Exploration** | Download, summary stats, missing values, visualizations, class balance | Loaded UC Irvine Dermatology dataset (34 clinical/histopathological features, 6 disease classes), handled missing age values, computed summary statistics, correlation heatmap, class distribution bar chart with clinical disease name labels |
| **Task 2: Data Preprocessing** | Target encoding, stratified 80/20 split, standardization | Label encoded 6-class target, `train_test_split` with `stratify=y`, `StandardScaler`, PCA and t-SNE 2D projections for class separability inspection |
| **Task 3: Model Training & Evaluation** | 3–5 ML classifiers, Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC | Implemented 6 classifiers: KNN, Multinomial Logistic Regression, SVM (RBF), Random Forest, Gradient Boosting, MLP Neural Network. Per-class and Macro Precision/Recall/F1, annotated confusion matrices, multiclass One-vs-Rest ROC-AUC curves for all classifiers |
| **Task 4: Advanced Concepts** | (Bonus — not explicitly required) | Hyperparameter grid search, SMOTE & class-weighted imbalance handling, feature importance biomarker extraction, interactive patient differential diagnosis simulator |

---

## Lab Assignment 6: CNN Models & Feature Extraction for Chest X-Ray Diagnosis

**Notebook:** [`Lab_Assignment_6_AI_in_Healthcare.ipynb`](Lab_Assignment_6_AI_in_Healthcare.ipynb)  
**Dataset:** Chest X-Ray Images (~5,863 images — Normal vs Pneumonia)

### What Was Asked

> Design and evaluate CNN models for automated chest disease diagnosis — download chest X-ray images, display samples, report class balance, resize to 128×128, apply HOG/LBP/GLCM feature extraction, prepare feature vectors, split 70/15/15, implement baseline CNN, tune hyperparameters with cross-validation, evaluate with Accuracy/Precision/Recall/F1/AUROC, confusion matrix with misclassification analysis, compare classifiers.

### What We Did

| Task | Requirement | Implementation |
|------|-------------|----------------|
| **Task 1: Data Acquisition & Exploration** | Download images, display 5–10 samples per class, report class balance | Loaded Chest X-Ray dataset (Normal vs Pneumonia), displayed sample radiographs with anatomical landmark annotations, cataloged file metadata (dimensions, aspect ratios), class distribution analysis |
| **Task 2: Data Preprocessing & Feature Extraction** | Resize to 128×128, grayscale, HOG, LBP, GLCM, feature vectors, 70/15/15 split | Resized all images to 128×128 grayscale, extracted HOG gradient orientation histograms, LBP uniform texture patterns (10-bin histogram), GLCM Haralick descriptors (Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM), assembled combined feature vectors, stratified 70/15/15 train-val-test split |
| **Task 3: Model Training & Evaluation** | Baseline CNN, hyperparameter tuning, cross-validation, all metrics, confusion matrix, classifier comparison | Built PyTorch CNN (3-stage Conv2D+BatchNorm+ReLU+MaxPool, AdaptiveAvgPool, FC classifier with Dropout), trained classical ML baselines (SVM, Random Forest, KNN, Logistic Regression), 5-Fold Stratified Cross-Validation with learning rate tuning, epoch-wise loss/accuracy learning curves, comprehensive test metrics (Accuracy, Precision, Recall, Specificity, F1, AUROC, AUPRC), annotated confusion matrix with misclassification image gallery, head-to-head classifier comparison identifying best performer |

---

## Lab Assignment 7: Deep Learning Grid Search with Scikit-Learn & Keras

**Notebook:** [`Lab_Assignment_7_AI_in_Healthcare.ipynb`](Lab_Assignment_7_AI_in_Healthcare.ipynb)  
**Dataset:** Pima Indians Diabetes (768 patients, 8 clinical features)

### What Was Asked

> Use the Grid-Search Capability of Scikit-Learn on a deep learning model. Using Keras models in Scikit-Learn, design a deep learning model and use GridSearchCV to tune:
> - Batch size and training epochs
> - Learning rate
> - Network weight initialization
> - Activation functions
> - Dropout regularization
> - Number of neurons in the hidden layer

### What We Did

| Task | Requirement | Implementation |
|------|-------------|----------------|
| **Task 1: Data Acquisition & Loading** | Load Pima Indians Diabetes dataset (as per previous labs) | Multi-tier dataset acquisition (local cache → GitHub → Plotly backup), zero-value clinical audit (Insulin 48.7%, SkinThickness 29.6% biologically impossible zeros), descriptive statistics, distribution KDE plots stratified by diabetes status with clinical reference cutoffs, Pearson correlation heatmap, class balance analysis (65.1% Non-Diabetic / 34.9% Diabetic) |
| **Task 2: Data Preprocessing** | Preprocess as per previous labs | Replaced biologically impossible zeros with NaN, KNN Imputer (k=5) multivariate imputation, stratified 80/20 train-test split preserving class ratio, StandardScaler fit strictly on training set, 2D PCA and t-SNE latent manifold projections |
| **Task 3: Grid Search — Batch Size & Epochs** | Tune batch size and training epochs | `GridSearchCV` with `KerasClassifier` wrapper, searched `batch_size ∈ {16, 32}` × `epochs ∈ {25, 50}`, 3-Fold Stratified CV, accuracy heatmap visualization |
| **Task 3: Grid Search — Learning Rate & Optimizer** | Tune learning rate | Searched `learning_rate ∈ {0.001, 0.005, 0.01}` × `optimizer ∈ {Adam, RMSprop}`, comparative bar chart |
| **Task 3: Grid Search — Weight Initialization** | Tune network weight initialization | Searched `init_mode ∈ {glorot_uniform, he_normal, uniform, normal}`, horizontal bar chart with std dev error bars |
| **Task 3: Grid Search — Activation Functions** | Tune activation functions | Searched `activation ∈ {relu, tanh, sigmoid, elu}`, comparative performance bar chart |
| **Task 3: Grid Search — Dropout Regularization** | Tune dropout regularization | Searched `dropout_rate ∈ {0.0, 0.1, 0.2, 0.3, 0.4}`, regularization bias-variance tradeoff curve with confidence band |
| **Task 3: Grid Search — Hidden Neurons** | Tune number of neurons in hidden layer | Searched `hidden_neurons ∈ {8, 16, 32, 64}`, model capacity bar chart |
| **Consolidated Evaluation** | (Bonus — beyond assignment scope) | Synthesized all 6 optimal hyperparameters into a single table, trained final optimized Keras model, plotted epoch loss/accuracy learning curves, evaluated on unseen test set with 4-panel diagnostic dashboard (Confusion Matrix, ROC Curve, Precision-Recall Curve, Calibration Reliability Diagram), head-to-head comparison vs un-tuned baseline model, interactive Clinical Decision Support System (CDSS) with 3 patient risk stratification simulations |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Data Processing | NumPy, Pandas, SciPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-Learn (Logistic Regression, SVM, KNN, Random Forest, Gradient Boosting, MLP) |
| Deep Learning | TensorFlow/Keras (Lab 7), PyTorch (Lab 6) |
| Hyperparameter Tuning | Scikit-Learn GridSearchCV + SciKeras KerasClassifier |
| Image Processing | Pillow, Scikit-Image (HOG, LBP, GLCM) |
| Signal Processing | WFDB (ECG), SciPy Signal |
| NLP | NLTK, Scikit-Learn TF-IDF |
| Imputation | Scikit-Learn KNNImputer |

---

*All notebooks include Hinglish code explanations for each cell.*
