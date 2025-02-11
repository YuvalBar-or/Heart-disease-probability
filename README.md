# Heart Disease Probabilty

This project is a machine learning-based web application designed to predict the likelihood of heart disease using different classification models. <br/>
In this project we aimed to see if simpler models like Decision Trees, Random Forest, and XGBoost could help in predicting heart disease. However, for the final app we designed, we used a more complex model using Neural Networks (NN). 

## The data: 
The dataset consists of several features relevant to heart disease prediction. Each of the following inputs plays a role in helping the model determine the likelihood of a patient having heart disease:

1. **Age**: The age of the patient in years. Older individuals are at a higher risk of heart disease, which makes age an important feature.
2. **Sex**: The gender of the patient (Male=1, Female=0). Gender can affect the likelihood of developing certain conditions, including heart disease.
3. **Chest Pain Type (cp)**: A categorical variable indicating the type of chest pain the patient is experiencing:
   - 1: Typical angina
   - 2: Atypical angina
   - 3: Non-anginal pain
   - 4: Asymptomatic
   Chest pain is often linked to heart problems, making this an essential feature in predicting heart disease.
4. **Resting Blood Pressure (trestbps)**: The resting blood pressure value of the patient in mm Hg. High blood pressure is a major risk factor for heart disease.
5. **Serum Cholesterol (chol)**: The cholesterol level in the blood in mg/dl. Elevated cholesterol levels are closely related to the risk of heart disease.
6. **Fasting Blood Sugar (fbs)**: This feature represents whether the patient's fasting blood sugar is greater than 120 mg/dl (1=True, 0=False). High fasting blood sugar levels can indicate diabetes, which increases the risk of heart disease.
7. **Resting Electrocardiographic Results (restecg)**: The result of an electrocardiogram that measures heart activity. Different values help determine if the heart is functioning properly.
8. **Maximum Heart Rate (thalach)**: The maximum heart rate achieved during exercise. Lower heart rates can indicate poor heart health.
9. **Exercise Induced Angina (exang)**: Whether the patient experiences angina during exercise (1=True, 0=False). Exercise-induced angina is a significant indicator of heart disease.
10. **Oldpeak (oldpeak)**: Depression induced by exercise relative to rest. A higher value is often associated with heart disease.
11. **Slope (slope)**: The slope of the peak exercise ST segment. This feature can provide insights into the heart's condition during exercise.
12. **Number of Major Vessels Colored by Fluoroscopy (ca)**: This refers to the number of blood vessels that were colored by fluoroscopy during the exam.
13. **Thalassemia (thal)**: A condition involving blood cells. It can impact heart health, making this an important feature for prediction.


## Models Used:
- **Decision Tree**: A basic yet effective machine learning model that splits the data into branches based on certain decision rules.
- **Random Forest**: An ensemble learning technique that uses multiple decision trees to improve the model's accuracy and reduce overfitting.
- **XGBoost**: A gradient boosting framework that focuses on improving the prediction accuracy through optimization techniques like regularization.

  While these models were explored to see if simpler models can work efficiently for heart disease prediction, the final app uses a Neural Network (NN) model that performed better on the data.


## Reason for Using Neural Networks in the App
While simpler models like Decision Trees, Random Forest, and XGBoost were initially explored to see if they could help in predicting heart disease, they had limitations in terms of performance and complexity. <br/>
A Neural Network was ultimately chosen for the app because it could capture complex patterns in the data, leading to better prediction results. Neural Networks are well-suited for this type of problem, where the relationships between variables are nonlinear and complex.

  
## Results: 
Risk Categories:
- Low Risk: <30%
- Medium Risk: 30–70%
- High Risk: >70%





