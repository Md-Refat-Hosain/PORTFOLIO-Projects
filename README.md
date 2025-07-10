
<img width="1536" height="1024" alt="ChatGPT Image Jul 10, 2025, 08_50_28 AM" src="https://github.com/user-attachments/assets/8efdb36f-443b-4ee4-9455-9f4c2f1f00a6" />
<br><br>

# Used car price prediction 
It is a comprehensive project.From data preprocessing to choose the best model (FastAPI is coming soon)

# Overview & Goal
- As I have said, it is an end-to-end data science project covering the nightmare of feature extraction from very messy text data, then feature engineering, EDA, model building, hyperparameter tuning, and finally choosing the best model.dataset is taken from 'kaggle'.

- My goal was simply in 2 points. 1. Can I extract insights from this messy dataset that are not visible to the naked eye, nor neither managers know about those insights, so that I can choke them with my explorations. 2. Can I help a newcomer in this business by making fairly accurate predictions, so that he can get a general price idea about the market and can become confident about the market? 

# Comprehensive Data Preprocessing 
### __1. Data Processing & Feature Engineering:__
<img width="1305" height="503" alt="Screenshot 2025-07-10 at 8 30 25 PM" src="https://github.com/user-attachments/assets/d639ef19-0c20-451f-bdd5-9f70d050de2b" />
<br><br>
__Source Data:__ The project utilized a comprehensive dataset of car listings, including attributes such as mileage, horsepower, age, transmission details, and various ca
 __Data Cleaning:__
- Handled missing values through appropriate imputation strategies (e.g., median for numerical features, mode for categorical features).

- Identified and addressed outliers in numerical features (ex: price, mileage, horsepower) to improve model robustness.

__Feature Engineering:__

- Created new features where applicable (e.g., Age from year of manufacture, if not directly available).

- Managed complex categorical features, including those with special characters like 'Plug-In Hybrid' and '–', ensuring they were correctly processed for model training.

__Data Transformation:__

- Applied log transformation to the target variable (car price) to mitigate skewness and stabilize variance, which significantly improved model performance.

- Used One-Hot Encoding for nominal categorical features and potentially Target Encoding / Label Encoding for high-cardinality features to convert them into a numerical format suitable for machine learning models.

- Standardization/Normalization was applied to numerical features to scale them consistently.
- (bewlo a glims of whole data preprocessing)
[<img width="1432" height="410" alt="Screenshot 2025-07-10 at 4 47 14 PM" src="https://github.com/user-attachments/assets/5701b36a-300a-43e5-b57c-fe8611c6abe3" />](
https://github.com/user-attachments/assets/25a890e4-4134-4ff6-b740-54a5cad48151)



  <br><br>

# ✨ Key Insights

### __Insigts from various continuous features__
<br><br>
<img width="1361" height="664" alt="Screenshot 2025-07-10 at 6 33 41 PM" src="https://github.com/user-attachments/assets/10c1d174-87b8-4d7d-b523-43d24d641632" />
<br><br>
<img width="1107" height="191" alt="Screenshot 2025-07-10 at 6 34 38 PM" src="https://github.com/user-attachments/assets/8dd70291-119a-4ad3-9ea2-5ef47cede962" />
### __Insigts from comparing bivariate features against price__
<br><br>
<img width="3780" height="1890" alt="Copy of Untitled Design" src="https://github.com/user-attachments/assets/6a4f038b-ee2a-4d03-9f90-028e9164e3d0" />
<br><br>
<img width="1339" height="210" alt="Screenshot 2025-07-10 at 6 40 10 PM" src="https://github.com/user-attachments/assets/f1df643c-3be7-476c-b327-f0edb0853632" />

<br><br>
### __Wihich sort of car got more sales__
<br><br>
<img width="1037" height="599" alt="Screenshot 2025-07-10 at 6 36 31 PM" src="https://github.com/user-attachments/assets/d90101ec-c42e-43e1-a6e4-bf283aa21d76" />
<br><br>

<img width="1373" height="213" alt="Screenshot 2025-07-10 at 8 21 22 PM" src="https://github.com/user-attachments/assets/8690410a-148b-43e1-b2a8-aca0492ba87f" />
<br><br>

### __Problamatic correlation detection among independent features__ 
<br><br>
<img width="306" height="622" alt="Screenshot 2025-07-10 at 6 41 45 PM" src="https://github.com/user-attachments/assets/66a4d0e0-3433-4365-809d-0ee74baf2023" />
<br><br>
<img width="1373" height="120" alt="Screenshot 2025-07-10 at 6 42 15 PM" src="https://github.com/user-attachments/assets/071bd246-0bd8-46f5-9262-48ce56a2193b" />
<br><br>
### __Outliers detection (Organic Outliers)__ 

<img width="1339" height="244" alt="Screenshot 2025-07-10 at 6 40 54 PM" src="https://github.com/user-attachments/assets/9f6eabef-165a-45f6-a829-b574d48d0ef0" />




# __Model Development & Evaluation__

- Model Selection: Explored a range of regression models to identify the best performer for the car price prediction task:

- Linear Regression: Established a baseline understanding of linear relationships.

- Random Forest Regressor: Leveraged ensemble learning for robust predictions and feature importance insights.

- XGBoost Regressor (Extreme Gradient Boosting): Employed a highly optimized gradient boosting framework known for its performance on tabular data.

__Hyperparameter Tuning:__ 
- Utilized GridSearchCV with K-Fold Cross-Validation to systematically search for the optimal hyperparameters for the best-performing model, ensuring robust generalization and preventing overfitting.

__Best Performing Model:__ 
- The Hyperparameter Tuned XGBoost Regressor emerged as the top performer, demonstrating superior predictive accuracy compared to other models.

__Evaluation Metrics:__ Model performance was rigorously assessed using:

- __R-squared (R2):__ 
- Measures the proportion of variance in the dependent variable that can be predicted from the independent variables. Achieved an R2 of 0.8072 (on log-transformed prices) with the best XGBoost model.

__Mean Absolute Error (MAE):__ Represents the average magnitude of the errors in a set of predictions, without considering their direction. Achieved an MAE of $17,641.91 (on original price scale).

__Root Mean Squared Error (RMSE):__ A measure of the differences between values predicted by a model and the values actually observed. The RMSE of $154,451.56 (on original price scale) indicated the presence of a few very large errors, primarily on high-value outlier cars, which was further investigated through residual analysis.

<img width="3780" height="1890" alt="Untitled design" src="https://github.com/user-attachments/assets/ecd0ea26-7b52-4d6c-931a-e6e9a5992ffe" />



# Deployment
Model Persistence: The final, best-performing XGBoost model was saved using joblib to facilitate its loading and use in a production environment.

__API Development with FastAPI:__

- Developed a RESTful API using FastAPI to serve real-time car price predictions.

- Defined a Pydantic BaseModel for input data validation, ensuring that incoming requests conform to the expected feature schema and data types. This robustly handles data integrity at the API endpoint.

- Implemented careful handling for feature names containing special characters (e.g., 'Plug-In Hybrid', '–') by mapping Pydantic's valid Python attribute names to the exact column names expected by the trained model.



__Deployment in Google Colab with ngrok:__

- The FastAPI application was deployed and run directly within a Google Colab environment.

- pyngrok was utilized to create a secure, publicly accessible tunnel to the local FastAPI server, enabling external access and testing of the API.

- threading was employed to run the uvicorn server in a separate thread, resolving event loop conflicts common in Jupyter/Colab environments and ensuring stable API operation.




- Interactive API Documentation: FastAPI automatically generated interactive API documentation (Swagger UI - accessible via /docs endpoint), allowing for easy testing and exploration of the /predict endpoint with example JSON inputs.

# Technical Aspect
- Programming Language: Python

- Data Manipulation: pandas, numpy

- Machine Learning: scikit-learn, xgboost

- Model Persistence: joblib

- API Development: FastAPI, uvicorn, pydantic

- Deployment Tunneling: pyngrok

- Visualization: matplotlib, seaborn


# __Future Enhancements & Learnings__
__Residual Analysis:__ A detailed residual analysis was performed to understand the nature of prediction errors, particularly the large discrepancies observed for high-value cars. This identified a need for more nuanced handling of outliers or rare, expensive vehicle types.


__Potential Improvements:__

- Further investigation into the features of high-error data points to identify unique characteristics or data quality issues.

- Exploring advanced feature engineering specific to luxury/exotic car segments.

- Consider a specialized sub-model or different modeling approach for very high-value vehicles if sufficient data for that segment is available.

- Implementing model monitoring (data drift, concept drift) post-deployment.

- Exploring model interpretability techniques (SHAP/LIME) for deeper insights.


