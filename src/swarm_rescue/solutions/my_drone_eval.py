from swarm_rescue.solutions.config import config

drone_eval_class = config["drone_eval_class"]


class MyDroneEval(drone_eval_class):
    """
    Evaluation drone class that inherits from MyDroneRL.

    This class can be extended to implement custom evaluation logic.
    """
    def __init__(self, **kwargs):
        kwargs['model_path'] = "src/swarm_rescue/solutions/models/a2c.zip"
        super().__init__(**kwargs)
