from datetime import timedelta

from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import IntervalSchedule

DeploymentSpec(
    flow_location="train.py",
    name="diabets_prediction",
    parameters={
        "raw_data_path": "data",
        "data_path": "data",
        "max_evals": 10,
    },
    schedule=IntervalSchedule(interval=timedelta(minutes=10)),
    flow_runner=SubprocessFlowRunner(),
    tags=['ml'],
)
