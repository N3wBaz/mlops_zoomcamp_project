** Problem description **
*Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high. Blood glucose is your main source of energy and comes from the food you eat. Insulin, a hormone made by the pancreas, helps glucose from food get into your cells to be used for energy.*

*This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.*

**Dataset’s variables:**

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
You need to create mlflow, model folders in repo, copy dataset diabetes.csv to data folder and following tools installed:
    • `pipenv`



## Preparation
Note: all actions expected to be executed in repo folder.
    • Install required packages ‘pipenv install --dev’
    • Run mlflow server with : ` mlflow server --host localhost --port 5000 --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root mlflow `
    • For single run start prefect with `prefect orion start` and then execute `python train.py’. It create prefect flow and run it. 
    • Run in another terminal prefect deployment:
`prefect deployment create train_deploy.py`
When it starts you should create work-queue at 127.0.0.1:4200, the copy agents script and run it in terminal. Agent start running and activate data preparation process, calculation best parameters of model, saving best model and registered it in mlflow every 10 minutes. Tracking experiment results and models parameters should see in http://127.0.0.1:5000, experiment name “Diabets-prediction-expriment”. 
