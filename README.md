# MLOps-ForcastSys

## ZenML setup steps

```bash
zenml init
zenml login --local
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

## Environment setup

```bash
pip install -r requirements.txt
```

## Training Pipeline
```bash
cd src/mlops/pipelines
python training_pipeline.py
```

## Deployment pipeline
```bash
cd src/
python app.py
```
- Check http://localhost:5555
