from swarm_rescue.solutions.my_drone_random import MyDroneRandom
from swarm_rescue.solutions.my_drone_RL import MyDroneRL

class MyDroneEval(MyDroneRL):
    """
    Evaluation drone class that inherits from MyDroneRL.

    This class can be extended to implement custom evaluation logic.
    """
    def __init__(self, **kwargs):
        kwargs['model_path'] = "models/ppo_drone_final.zip"
        super().__init__(**kwargs)
    pass