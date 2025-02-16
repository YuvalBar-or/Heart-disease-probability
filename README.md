# Heart Disease Probabilty

 #### Think your heart’s just fine? Let’s put it to the test! <br/>
https://heart-disease-probability-9ivfcw5g2epgartph9ek8r.streamlit.app/

## The App : 
Upon logging into the app, you'll be greeted with a brief explanation of how it works:
<img width="692" alt="Screenshot 2025-02-12 at 1 46 00" src="https://github.com/user-attachments/assets/ae2b243d-9aca-487b-88d4-842b1495fb7e" />

Next, you’ll be prompted to answer a series of questions (some may require assistance from a healthcare professional or specific medical tests). <br/>
Once completed, the app will process all your inputs and provide you with a probability result indicating your likelihood of having heart disease. <br/> 
GOOD LUCK ! and may the odds be ever in your favor ! 

Risk Categories:
- Low Risk: <30%
- Medium Risk: 30–70%
- High Risk: >70%

## About the project:
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

### Lets have a quick look at the data: 
<img width="669" alt="Screenshot 2025-02-12 at 1 18 29" src="https://github.com/user-attachments/assets/97be917a-96e8-4bbe-abaf-60379d88692d" />
<img width="969" alt="Screenshot 2025-02-12 at 1 20 20" src="https://github.com/user-attachments/assets/fd6006b0-ad16-4e2d-8939-5d90ddcdf84a" />
<img width="583" alt="Screenshot 2025-02-12 at 1 20 29" src="https://github.com/user-attachments/assets/236d6c56-7052-42c0-846f-a004071f8acf" />

<br/>
let's look at the distribution of all the features in the data: 
<br/>
<br/>
<br/>
<img width="580" alt="Screenshot 2025-02-12 at 1 24 17" src="https://github.com/user-attachments/assets/cb407d35-e4b7-4bd5-85b4-23b0b121c6e7" />
<img width="565" alt="Screenshot 2025-02-12 at 1 24 24" src="https://github.com/user-attachments/assets/b1494467-6c8a-4ac8-8707-a32686352ea1" />
<img width="562" alt="Screenshot 2025-02-12 at 1 25 05" src="https://github.com/user-attachments/assets/eed27007-0d0f-4cec-818a-fa4fddaba840" />
<img width="574" alt="Screenshot 2025-02-12 at 1 25 13" src="https://github.com/user-attachments/assets/2136b7d3-6365-4f1a-92e1-332d7655b55e" />
<img width="573" alt="Screenshot 2025-02-12 at 1 24 10" src="https://github.com/user-attachments/assets/9d41e454-fc0b-49b2-8539-e7f7ddf0acb7" />
<img width="565" alt="Screenshot 2025-02-12 at 1 24 37" src="https://github.com/user-attachments/assets/c56ca9ee-ec8b-4227-a701-4469e15415c8" />
<img width="562" alt="Screenshot 2025-02-12 at 1 24 30" src="https://github.com/user-attachments/assets/0fef8572-824f-4345-9d72-6cde1b8300c7" />
<img width="569" alt="Screenshot 2025-02-12 at 1 25 32" src="https://github.com/user-attachments/assets/3971dc60-b5a1-4467-a7e6-4b89f4803b57" />
<img width="568" alt="Screenshot 2025-02-12 at 1 25 39" src="https://github.com/user-attachments/assets/b8f1e9ea-6b3e-4217-9b5e-64391c05530f" />
<img width="567" alt="Screenshot 2025-02-12 at 1 25 45" src="https://github.com/user-attachments/assets/d692c68b-243d-4b00-af26-146c72eb04fe" />
<img width="570" alt="Screenshot 2025-02-12 at 1 25 55" src="https://github.com/user-attachments/assets/0cfdb880-8e5a-49a4-9275-0188ab8e81c2" />
<img width="573" alt="Screenshot 2025-02-12 at 1 26 02" src="https://github.com/user-attachments/assets/db19b842-4c00-4036-a96d-b8147c8c547f" />
<img width="578" alt="Screenshot 2025-02-12 at 1 26 08" src="https://github.com/user-attachments/assets/f71bc6e0-141b-4ec7-b33c-252389df0bac" />


## Models Used:
- **Decision Tree**: A basic yet effective machine learning model that splits the data into branches based on certain decision rules.
- **Random Forest**: An ensemble learning technique that uses multiple decision trees to improve the model's accuracy and reduce overfitting.
- **XGBoost**: A gradient boosting framework that focuses on improving the prediction accuracy through optimization techniques like regularization.

  While these models were explored to see if simpler models can work efficiently for heart disease prediction, the final app uses a Neural Network (NN) model that performed better on the data.

### Results for models: 
#### Decision Tree:
<img width="438" alt="Screenshot 2025-02-12 at 1 29 18" src="https://github.com/user-attachments/assets/b79fb541-829c-43ce-a317-740f5b1ddcc9" />
 
 
 we even looked into the tree to try and analyze the decisions in each node: 
 
<img width="543" alt="Screenshot 2025-02-12 at 1 29 06" src="https://github.com/user-attachments/assets/a1873a36-9aec-4e5a-9469-3c22f6755bbc" />

#### RandomForest: 
<img width="431" alt="Screenshot 2025-02-12 at 1 30 52" src="https://github.com/user-attachments/assets/c00ab244-cd28-4118-91fd-4d5aa17750ec" />

we even looked at the feature importance to try and understand the decision making of that model: 
<img width="574" alt="Screenshot 2025-02-12 at 1 31 03" src="https://github.com/user-attachments/assets/e33c2da7-4675-449a-a83d-0dbdb8d49806" />

#### XGBOOST:
<img width="437" alt="Screenshot 2025-02-12 at 1 32 12" src="https://github.com/user-attachments/assets/b4d3408b-b4cd-42b3-92b4-9ee81477cd91" />


we even looked at the feature importance to try and understand the decision making of that model: 
<img width="576" alt="Screenshot 2025-02-12 at 1 32 18" src="https://github.com/user-attachments/assets/45c97987-3564-4aa5-b8a0-4d3a8d753a3a" />

#### NN: 
for this model, we ran it 3 times : 15 epochs , 30 epochs and 60 epochs. 

##### 15 epochs: 
<img width="428" alt="Screenshot 2025-02-12 at 1 35 29" src="https://github.com/user-attachments/assets/c2741833-afa8-4d66-9a8e-f78b806b642b" />
<img width="639" alt="Screenshot 2025-02-12 at 1 35 37" src="https://github.com/user-attachments/assets/d04987bc-6f05-467f-8e11-f5b0ae7f8d56" />

##### 30 epochs: 
<img width="427" alt="Screenshot 2025-02-12 at 1 36 10" src="https://github.com/user-attachments/assets/10aa1832-ebbb-45ff-addf-47f06ad9f4bc" />
<img width="640" alt="Screenshot 2025-02-12 at 1 36 17" src="https://github.com/user-attachments/assets/1ed38e29-5a6c-40a9-bedd-279c68b99de3" />


##### 60 epochs: 
<img width="435" alt="Screenshot 2025-02-12 at 1 37 04" src="https://github.com/user-attachments/assets/82feb028-2278-4e14-88b1-ae0581975b39" />
<img width="646" alt="Screenshot 2025-02-12 at 1 37 11" src="https://github.com/user-attachments/assets/c2308df3-8ad9-44cc-9642-435a3c1c78d1" />


## Reason for Using Neural Networks in the App
While simpler models like Decision Trees, Random Forest, and XGBoost were initially explored to see if they could help in predicting heart disease, they had limitations in terms of performance and complexity. <br/>
A Neural Network was ultimately chosen for the app because it could capture complex patterns in the data, leading to better prediction results. Neural Networks are well-suited for this type of problem, where the relationships between variables are nonlinear and complex.

### Conclusion
This project focused on predicting the likelihood of heart disease using machine learning models, leveraging medical and demographic factors such as age, cholesterol levels, and blood pressure. <br/>
The models tested included Decision Tree, Random Forest, XGBoost, and Neural Networks. The results demonstrated that age, sex, cholesterol levels, and blood pressure are key predictors of heart disease, consistent with prior research on cardiovascular risk factors. Among the models tested, Neural Networks, especially when trained for 60 epochs, achieved the highest accuracy (0.92) and AUC (0.95), outperforming other models in both precision and recall. This confirms the Neural Network's ability to capture complex, non-linear relationships between features, making it a powerful tool for medical predictions. Random Forest and XGBoost also showed strong performance, with improved AUC and predictive reliability compared to Decision Trees. The ability of these models to identify key risk factors is critical for accurate heart disease prediction. <br/>
The findings suggest that machine learning models, particularly Neural Networks, can significantly aid healthcare providers in early detection and prevention of heart disease. <br/>
Future work will focus on enhancing the app’s usability, integrating additional data sources, and further refining the models for even better predictive accuracy. Ultimately, this project aims to provide healthcare providers with a reliable tool to identify individuals at risk for heart disease and offer early interventions that could prevent life-threatening events.








  






