import mlflow

class ModelLoader(object):

    @staticmethod
    def load_model(model_name, run_id):
        # Load the logged model using the run ID
        model_uri = f"runs:/{run_id}/{model_name}"
        model = mlflow.lightgbm.load_model(model_uri)
        return model, model_uri