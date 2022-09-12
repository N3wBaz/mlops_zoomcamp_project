import os
import pickle
import argparse

import pandas as pd
from prefect import flow, task
from hyperopt import STATUS_OK, Trials, hp, tpe, fmin
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from prefect.task_runners import SequentialTaskRunner

import mlflow
from preprocess_data import preprocess_run

# from hyperopt.pyll import scope


def dump_pickle(obj, filename):
    with open(filename, 'wb') as file_out:
        return pickle.dump(obj, file_out)


def load_data(filename: str):
    df = pd.read_csv(filename)
    target = "Outcome"
    y = df[target]
    df = df.drop(columns=["Outcome"], axis=1)
    return df, y


@task
def preprocess_data(raw_data_path: str, data_path: str):
    raw_data_path = "data"
    preprocess_run(raw_data_path, data_path)


@task
def hpo(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
    num_trials: int,
):
    # pylint: disable=unused-variable

    # search params with HPO
    def objective(params):

        with mlflow.start_run():

            mlflow.set_tag("developer", "ruslan")
            mlflow.set_tag("model", "random forest")
            mlflow.log_params(params)
            rf = RandomForestClassifier(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_valid)
            acc = accuracy_score(y_valid, y_pred)
            mlflow.log_metric("accuracy", acc)

        return {"loss": -acc, "status": STATUS_OK}

    search_space = {
        "n_estimators": hp.randint("n_estimators", 50, 100),
        "max_depth": hp.randint("max_depth", 5, 100),
        "min_samples_split": hp.uniform("min_samples_split", 0, 1),
        "min_samples_leaf": hp.randint("min_samples_leaf", 1, 10),
        "random_state": 42,
    }
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        trials=Trials(),
        max_evals=num_trials,
    )
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)
    dump_pickle((X_valid, y_valid), os.path.join('model', "last_model.bin"))


# ---------------------------------------------------
#       Choose the best metric's model
#       save it, write parameters to mlflow
#       and register.
# ---------------------------------------------------
@task
def train_best_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
):

    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids="1",
        filter_string="",
        order_by=["metrics.accuracy DESC"],
        max_results=3,
    )
    print("3 models with best accuracy:")
    for run in runs:
        print(
            f"run id: {run.info.run_id}, accuracy: {run.data.metrics['accuracy']:.4f}"
        )
    best_run = runs[0]

    best_params = {
        i[0]: float(i[1]) if "." in i[1] else int(i[1])
        for i in best_run.data.params.items()
    }
    model = RandomForestClassifier(**best_params)

    mlflow.sklearn.autolog(disable=True)

    with mlflow.start_run() as run:

        mlflow.log_params(best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        acc = accuracy_score(y_valid, y_pred)
        mlflow.log_metric("accuracy", acc)
        mlflow.set_tag("status", "best model")
        mlflow.set_tag("developer", "ruslan")
        mlflow.set_tag("model", "random forest")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="models_mlflow",
            registered_model_name="sk-learn-rf-cl-model",
        )


@flow(task_runner=SequentialTaskRunner())
def diabets_prediction(raw_data_path: str, data_path: str, max_evals: int):

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Diabets-prediction-expriment")

    preprocess_data(raw_data_path, data_path)
    X_train, y_train = load_data(os.path.join('data', "train.csv"))
    X_valid, y_valid = load_data(os.path.join('data', "valid.csv"))

    hpo(X_train, y_train, X_valid, y_valid, max_evals)
    train_best_model(X_train, y_train, X_valid, y_valid)


def runner(raw_data_path, data_path, max_evals):
    diabets_prediction(raw_data_path, data_path, max_evals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_path",
        default="data",
        help="the location where the raw diabets data was saved",
    )
    parser.add_argument(
        "--data_path",
        default="data",
        help="the location where the processed diabets data was saved.",
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=10,
        help="the number of parameter evaluations for the optimizer to explore.",
    )

    args = parser.parse_args()
    runner(args.raw_data_path, args.data_path, args.max_evals)
