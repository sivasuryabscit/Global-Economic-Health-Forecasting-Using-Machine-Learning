# Global-Economic-Health-Forecasting-Using-Machine-Learning

# **PROJECT DOCUMENTATION**

## **STAGE 1: Initial Exploratory Data Analysis (EDA)**
**Objective:** To get acquainted with the raw dataset.

**Key Activities:**
- **Data Loading & Inspection:** Used `df.info()`, `df.describe()`, `df.shape`, and `df.columns` to explore data structure and quality.
- **Initial Understanding:** Gained foundational knowledge of feature types, value distributions, and dataset size.

---

## **STAGE 2: EDA (Visualization) & Pre-processing**
**Objective:** Clean, transform, and visualize data for better insights and readiness for classification modeling.

**Key Activities:**
- **Handling Missing Values:** Removed rows with missing target values and used median imputation (both country-specific and global) for others.
- **Handling Duplicates:** Identified and eliminated duplicates to ensure integrity.
- **Outlier Detection & Treatment:** Applied the IQR method to detect and cap outliers in feature columns.
- **Skewness Check & Transformation:** Used `log1p`/`sqrt` to normalize skewed variables.
- **Post-Preprocessing EDA:** Validated data quality using `.info()`, `.describe()`, `.shape`.
- **Visualizations:**
  - Univariate histograms
  - Bivariate scatter plots
  - Multivariate pair plots
  - Boxplots (before & after outlier treatment)
  - Skewness plots (before & after transformation)

---

## **STAGE 3: Feature Selection & Classification Model Building**
**Objective:** Prepare the dataset and train initial classification models.

**Key Activities:**
- **Feature Engineering:**
  - `inflation_unemployment`
  - `debt_to_revenue`
  - `income_group_encoded`
- **Feature Selection:**
  - **Chi-Squared Test**
  - **Correlation Matrix & ANOVA F-test**
- **Train-Test Split:** Stratified split for classification tasks.
- **Model Building – Classification:**
  - Decision Tree Classifier
  - Random Forest Classifier
  - k-Nearest Neighbors (k-NN)

---

## **STAGE 4: Model Evaluation & Comparison**

**Objective:**

Model Evaluation:
Calculated standard classification metrics to evaluate model performance, including:

* Accuracy

* Precision

* Recall

* F1-Score

* Confusion Matrix

Classification Report

Model Comparison:
Summarized and compared the performance of all classification models using organized DataFrames. The goal was to identify the best-performing classification model based on evaluation metrics and cross-validation results.

Model Selection Post-Tuning:
After hyperparameter tuning using GridSearchCV, the Random Forest Classifier showed the best performance and was selected as the final model for deployment.

###  Model Comparison Summary (Post-Hyperparameter Tuning)

| Model                  | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Decision Tree          | 0.5827   | 0.6061    |   -    |    -     |
| k-Nearest Neighbors    | 0.5482   | 0.5603    |   -    |    -     |
| **Random Forest (Tuned)** | **0.6561** | **0.6743** | **0.6561** | **0.6575** |

---

###  Best Parameters from GridSearchCV

```python
 param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
