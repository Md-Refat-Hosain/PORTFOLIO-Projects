
<img width="700" height="400" alt="ChatGPT Image Jul 10, 2025, 08_50_28 AM (1)" src="https://github.com/user-attachments/assets/4f93e3b7-2f02-4a20-9cc8-43ec742eeff2" />

# Used Car Price Prediction

This is a comprehensive, end-to-end data science project demonstrating the full lifecycle of building a machine learning model for used car price prediction, from handling messy raw data to robust model development.

## 🎯 Overview & Goal

* This project represents a complete end-to-end data science endeavor, tackling the complexities of feature extraction from highly unstructured text data. It encompasses robust data preprocessing, in-depth Exploratory Data Analysis (EDA), rigorous model building, advanced hyperparameter tuning, and the final selection of the optimal predictive model. The dataset was sourced from Kaggle.

* My primary objectives for this project were two-fold:
    1.  To uncover and extract non-obvious, actionable insights from this challenging dataset that could provide novel value to stakeholders and inform strategic decisions.
    2.  To develop a highly accurate and reliable prediction model that can empower new entrants in the used car market by providing clear market price estimations, thereby fostering confidence and informed decision-making.

## 🚀 Project Structure & How to Explore

To understand and explore this project, navigate through the following key sections and files in the repository:

### 1. Data & Notebooks

* **`data/`**: This directory contains the datasets used in the project.
    * `data/raw/`: Stores the original, untouched dataset.
    * `data/processed/`: Contains the cleaned and preprocessed data ready for modeling.
* **`notebooks/`**: This folder houses the Jupyter Notebooks detailing each stage of the project.
    * `01_EDA.ipynb`: Comprehensive Exploratory Data Analysis, including all visualizations and initial insights.
    * `02_Preprocessing.ipynb`: Detailed steps for data cleaning, feature engineering, and transformation.
    * `03_Model_Training_Evaluation.ipynb`: Model selection, initial training, and performance evaluation.
    * `04_Hyperparameter_Tuning.ipynb`: Advanced optimization of the best model's hyperparameters.
    * `05_Residual_Analysis.ipynb`: In-depth analysis of model prediction errors.

### 2. Model Artifacts

* **`models/`**: This directory stores the trained machine learning models.
    * `best_xgboost_car_price_model.joblib`: The final, best-performing XGBoost model saved for future use.

### 3. Supporting Files

* **`requirements.txt`**: Lists all the Python libraries and their exact versions required to run this project, ensuring environment reproducibility.
* **`images/`**: Contains all the plots and screenshots embedded in this `README.md` for visual context.

## ✨ Key Insights

This section presents the key findings and visualizations from the Exploratory Data Analysis (EDA) that informed the modeling approach.

### Insights from Various Continuous Features

* This plot showcases the distribution of key continuous features (e.g., price, mileage, horsepower), revealing their spread and potential skewness. Understanding these distributions is crucial for effective data transformation.
    ![Continuous Features Insights](https://github.com/user-attachments/assets/10c1d174-87b8-4d7d-b523-43d24d641632)
    ![Continuous Features Insights Text](https://github.com/user-attachments/assets/8dd70291-119a-4ad3-9ea2-5ef47cede962)

### Insights from Comparing Bivariate Features Against Price

* These visualizations explore the relationships between individual features and the target variable (car price), helping to identify features with strong predictive power and patterns.
    ![Bivariate Features vs Price](https://github.com/user-attachments/assets/6a4f038b-ee2a-4d03-9f90-028e9164e3d0)
<img width="1339" height="210" alt="Screenshot 2025-07-10 at 6 40 10 PM" src="https://github.com/user-attachments/assets/ded0b2d2-b0a9-4cd6-bc42-a2cc8ca363cf" />




### Which Sort of Car Got More Sales

* This analysis identifies the most popular car types or segments in the dataset based on sales volume, providing valuable market insights into consumer preferences.
    ![Car Sales Distribution](https://github.com/user-attachments/assets/d90101ec-c42e-43e1-a6e4-bf283aa21d76)
    ![Car Sales Distribution Text](https://github.com/user-attachments/assets/8690410a-148b-43e1-b2a8-aca0492ba87f)

### Problematic Correlation Detection Among Independent Features

* This section highlights the detection of multicollinearity among independent features using a correlation matrix, which can impact model stability and interpretability.

  <img width="826" height="435" alt="Screenshot 2025-07-12 at 11 34 50 AM" src="https://github.com/user-attachments/assets/976d4b2e-b4cd-4d9a-8123-5e502f0b9298" />

    ![Correlation Matrix Text](https://github.com/user-attachments/assets/071bd246-0bd8-46f5-9262-48ce56a2193b)

### Outliers Detection (Organic Outliers)

* This visualization helps in identifying natural (organic) outliers within the dataset, which are extreme but valid data points, distinct from data entry errors, and require careful handling.
<img width="1339" height="244" alt="Screenshot 2025-07-10 at 6 40 54 PM" src="https://github.com/user-attachments/assets/c04920b3-a05b-4923-b5b6-0e494a4a2679" />

## ⚙️ Technical Aspects

This section details the technical methodologies and tools employed throughout the Car Price Prediction project, from data ingestion to model development.


(Before data preprocessing snapshot)
<img width="2362" height="1181" alt="Untitled design (1)" src="https://github.com/user-attachments/assets/83019830-fc2d-403b-8c22-d586e70d0cd9" />




### 1. Data Processing & Feature Engineering

* **Source Data:** The project utilized a comprehensive dataset of car listings, including attributes such as mileage, horsepower, age, transmission details, and various categorical features.


### Data Cleaning:

* Handled missing values through appropriate imputation strategies (e.g., median for numerical features, mode for categorical features).

* Identified and addressed outliers in numerical features (e.g., price, mileage, horsepower) to improve model robustness.

### Feature Engineering:

* Created new features where applicable (e.g., Age from year of manufacture, if not directly available).

* Managed complex categorical features, including those with special characters like 'Plug-In Hybrid' and '–', ensuring they were correctly processed for model training.

### Data Transformation:

* Applied log transformation to the target variable (car price) to mitigate skewness and stabilize variance, which significantly improved model performance.

* Used One-Hot Encoding for nominal categorical features and potentially Target Encoding / Label Encoding for high-cardinality features to convert them into a numerical format suitable for machine learning models.

* Standardization/Normalization was applied to numerical features to scale them consistently.
* Below is a glimpse of the whole data preprocessing:
    [<img width="1432" height="410" alt="Screenshot 2025-07-10 at 4 46 42 PM" src="https://github.com/user-attachments/assets/3813c89d-47a0-4ff8-9bb0-9c200d15eebc" />](https://youtu.be/sf5lNF4LPTYsi=dHbAVIKeAEd46awW)


## ⚙️ Technical Aspects: Model Development & Evaluation

### Model Development & Evaluation

* **Model Selection:** Explored a range of regression models to identify the best performer for the car price prediction task:
    * **Linear Regression:** Established a baseline understanding of linear relationships.
    * **Random Forest Regressor:** Leveraged ensemble learning for robust predictions and feature importance insights.
    * **XGBoost Regressor (Extreme Gradient Boosting):** Employed a highly optimized gradient boosting framework known for its performance on tabular data.

### Hyperparameter Tuning:

* Utilized GridSearchCV with K-Fold Cross-Validation to systematically search for the optimal hyperparameters for the best-performing model, ensuring robust generalization and preventing overfitting.

### Best Performing Model:

* The Hyperparameter Tuned XGBoost Regressor emerged as the top performer, demonstrating superior predictive accuracy compared to other models.

### Evaluation Metrics: Model performance was rigorously assessed using:

* **R-squared ($R^2$):**
    * Measures the proportion of variance in the dependent variable that can be predicted from the independent variables. Achieved an $R^2$ of **0.8072 (on log-transformed prices)** with the best XGBoost model.

* **Mean Absolute Error (MAE):** Represents the average magnitude of the errors in a set of predictions, without considering their direction. Achieved an MAE of **\$17,641.91 (on original price scale)**.

* **Root Mean Squared Error (RMSE):** A measure of the differences between values predicted by a model and the values actually observed. The RMSE of **\$154,451.56 (on original price scale)** indicated the presence of a few very large errors, primarily on high-value outlier cars, which was further investigated through residual analysis.
    ![Model Performance Metrics](https://github.com/user-attachments/assets/ecd0ea26-7b52-4d6c-931a-e6e9a5992ffe)

## 🛠️ Tools & Libraries

* **Programming Language:** Python

* **Data Manipulation:** `pandas`, `numpy`

* **Machine Learning:** `scikit-learn`, `xgboost`

* **Model Persistence:** `joblib`

* **Visualization:** `matplotlib`, `seaborn`

* **Version Control:** `Git`, `GitHub`

## 🚀 Future Enhancements & Learnings

### Residual Analysis:

* A detailed residual analysis was performed to understand the nature of prediction errors, particularly the large discrepancies observed for high-value cars. This identified a need for more nuanced handling of outliers or rare, expensive vehicle types.

### Potential Improvements:

* Further investigation into the features of high-error data points to identify unique characteristics or data quality issues.

* Exploring advanced feature engineering specific to luxury/exotic car segments.

* Consider a specialized sub-model or different modeling approach for very high-value vehicles if sufficient data for that segment is available.

* Implementing model monitoring (data drift, concept drift) post-deployment.

* Exploring model interpretability techniques (SHAP/LIME) for deeper insights.
