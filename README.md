# MLops-zoomcamp project
### Problem description
Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high. Blood glucose is your main source of energy and comes from the food you eat. Insulin, a hormone made by the pancreas, helps glucose from food get into your cells to be used for energy.

This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

#### Dataset’s variables:

* Pregnancies: The number of pregnancies
* Glucose: The plasma glucose concentration in the oral glucose tolerance test after two hours
* Blood Pressure: Blood Pressure (Small blood pressure) (mmHg)
* SkinThickness: Skin Thickness
* Insulin: 2-hour serum insulin (mu U/ml)
* DiabetesPedigreeFunction: This function calculates the likelihood of having diabetes based on the lineage of a descendant
* BMI: Body mass index
* Age: Age (year)
* Outcome: Have the disease (1) or not (0)
## Prerequisites
* Create mlflow folder, model folders in repo (if there are none), copy dataset  [Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database "Pima Indians Diabetes Database") with name `diabetes.csv` to data folder.
* Install pipenv, if need run in terminal `pip install pipenv`

## Preparation

* All actions expected to be executed in repo folder.
* Install required packages with `pipenv install --dev` or `make setup` in terminal.
* Activate virtual enviromnent with `pipenv shell`
* For next step run mlflow server with following command: 
    ```bash
       mlflow server --host localhost --port 5000 --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root mlflow
    ```
* For single run start prefect with: 
    ``` bash 
       prefect orion start
    ```
* And if then execute `python train.py` in terminal. It create prefect flow and run it once. 
* For execute prefect deployment with scheduler execute in another terminal following command:
    ```bash
       prefect deployment create train_deploy.py
    ```
When deployment starts create *work-queue* at Prefect Orion server in `http://127.0.0.1:4200`, then copy orion agents script and run it in terminal. Agent start running scheduler and activate data preparation process, calculation best parameters of model, saving best model and registered it in mlflow every 10 minutes. Tracking experiments results and models parameters should see at Mlflow server in `http://127.0.0.1:5000`, experiment name **Diabets-prediction-expriment**. 
