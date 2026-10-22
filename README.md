\# Customer Churn Prediction



A machine learning project that predicts whether a telecommunications customer is likely to leave a service provider.



\## Project Overview



Customer churn occurs when customers stop using a company's services. This project uses customer information and service details to predict churn and help identify customers who may need additional attention.



\## Objectives



\* Explore and understand customer churn data.

\* Clean and preprocess the dataset.

\* Perform exploratory data analysis.

\* Train and evaluate machine learning models.

\* Build an interactive web application for churn predictions.



\## Technologies Used



\* Python

\* Pandas

\* Scikit-learn

\* Matplotlib

\* Joblib

\* Streamlit



\## Machine Learning Models



Two models were evaluated:



| Model               | Accuracy | ROC-AUC |

| ------------------- | -------: | ------: |

| Logistic Regression |   80.38% |  0.8357 |

| Random Forest       |   75.05% |  0.8359 |



The Random Forest model was selected for the application because it achieved higher recall for identifying customers who churned.



\## Project Structure



```text

customer-churn-prediction/

├── app.py

├── data\_exploration.py

├── requirements.txt

├── data/

│   ├── WA\_Fn-UseC\_-Telco-Customer-Churn.csv

│   └── cleaned\_churn\_data.csv

└── src/

&#x20;   ├── data\_cleaning.py

&#x20;   ├── data\_preprocessing.py

&#x20;   ├── eda.py

&#x20;   ├── evaluate\_model.py

&#x20;   ├── train\_model.py

&#x20;   ├── train\_random\_forest.py

&#x20;   └── random\_forest\_model.pkl

```



\## How to Run



\### 1. Clone the repository



```bash

git clone <your-repository-url>

cd customer-churn-prediction

```



\### 2. Create a virtual environment



```bash

python -m venv .venv

```



\### 3. Activate the environment



Windows Command Prompt:



```cmd

.venv\\Scripts\\activate.bat

```



\### 4. Install dependencies



```bash

python -m pip install -r requirements.txt

```



\### 5. Launch the application



```bash

python -m streamlit run app.py

```



The application will open in your browser.



\## Limitations



\* Predictions depend on the available dataset and the model's learned patterns.

\* Model predictions are estimates and should not be treated as guarantees.

\* Further validation is needed before using the model for real business decisions.



\## Future Improvements



\* Improve model performance through hyperparameter tuning.

\* Add visualizations to the web application.

\* Explore additional classification algorithms.

\* Deploy the application online.



\## Author



Dipanwita Das

B.Tech, Data Science

Heritage Institute of Technology





