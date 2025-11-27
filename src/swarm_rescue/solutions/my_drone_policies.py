'''
Definitons of custom drone policies. (RL based loaded from model or hardcoded logical policies)
'''
class RandomPolicy:
    """
    Random policy generating actions compatible with DroneRLEnv.
    Returns a (4,) numpy array:
        [forward, lateral, rotation, grasper]
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, state, **kwargs):
        # Sample from the action space directly (cleanest way)
        return self.action_space.sample()
