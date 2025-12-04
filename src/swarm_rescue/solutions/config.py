'''
Config for the drone challenge
'''
from swarm_rescue.solutions.my_drone_FSM import MyDroneFSM
from swarm_rescue.solutions.my_drone_random import MyDroneRandom
from swarm_rescue.solutions.RL.my_drone_RL import MyDroneRL

config = {
    "drone_eval_class": MyDroneFSM, # Choose from [MyDroneRL, MyDroneRandom, MyDroneFSM]
}