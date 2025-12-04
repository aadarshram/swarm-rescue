'''
Drone Controller with Finite State Machine (FSM) behavior. Uses programmed logic for selecting actions and also the actions themselves.
'''

import math
import random
from enum import Enum
from typing import Optional

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.utils.utils import normalize_angle


class MyDroneFSM(DroneAbstract):
    
    class State(Enum):
        """
        FSM States
        """
        SEARCHING_WOUNDED = 1
        GRASPING_WOUNDED = 2
        DROPPING_RESCUE_CENTER = 3
        RETURNING_TO_BASE = 4

    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        """
        Initializes the FSM drone controller.

        Args:
            identifier (Optional[int]): Drone identifier.
            misc_data (Optional[MiscData]): Miscellaneous data.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)
        
        # Initialize FSM state
        self.state = self.State.SEARCHING_WOUNDED

        # Save initial position (rescue center is usually near start)
        self.start_position = None
        self.return_base = None # Same as start position
        self.rescue_center_position = None
        self.found_rescue_center = False
        
        # Communication: which wounded person this drone is targeting
        self.target_wounded_position = None
        self.claimed_wounded_positions = {}  # {drone_id: wounded_position}
        
        # Random exploration fallback
        self.counterStraight = 0
        self.angleStopTurning = random.uniform(-math.pi, math.pi)
        self.distStopStraight = random.uniform(10, 50)
        self.isTurning = False

    def define_message_for_all(self):
        """
        Communication between drones controlled by same FSM.
        Message format: (drone_id, (state, rescue_center_pos, target_wounded_pos, current_pos))
        """
        current_pos = self.measured_gps_position()
        msg_data = (
            self.identifier,
            (
                self.state.value,  # Current state
                self.rescue_center_position if self.found_rescue_center else None,
                self.target_wounded_position,  # Which wounded person I'm targeting
                current_pos  # My current position for swarm spreading TODO
            )
        )
        return msg_data
    
    def process_communication(self):
        """
        Process messages from other drones to:
        1. Learn rescue center location from others
        2. Know which wounded persons are claimed by other drones
        3. Track other drones' positions to avoid clustering
        """
        if not self.communicator:
            return
        
        received_messages = self.communicator.received_messages
        
        # Clear old data
        self.claimed_wounded_positions.clear()
        
        for msg in received_messages:
            message = msg[1]
            other_drone_id = message[0]
            other_state, other_rescue_pos, other_target_wounded, other_position = message[1]
            
            # Learn rescue center location from other drones
            if other_rescue_pos is not None and not self.found_rescue_center:
                self.rescue_center_position = other_rescue_pos
                self.found_rescue_center = True
                print(f"Drone {self.identifier}: Learned rescue center location from Drone {other_drone_id}!")
            
            # Track which wounded persons are claimed by other drones
            if other_target_wounded is not None:
                self.claimed_wounded_positions[other_drone_id] = other_target_wounded

    
    def is_wounded_claimed(self, wounded_position, tolerance=50.0):
        """
        Check if a wounded person at the given position is already claimed by another drone.
        If multiple drones claim the same target, randomly assign one and others back off.
        
        Args:
            wounded_position: (x, y) position of the wounded person
            tolerance: Distance threshold to consider positions as "same" wounded person
            
        Returns:
            bool: True if this drone should back off, False if it can proceed
        """
        if wounded_position is None:
            return False
        
        # Check if any other drone is targeting the same wounded person
        competing_drones = [self.identifier]  # Include self
        
        for other_drone_id, other_target in self.claimed_wounded_positions.items():
            if other_target is None:
                continue
            
            # Calculate distance between target positions
            dx = wounded_position[0] - other_target[0]
            dy = wounded_position[1] - other_target[1]
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < tolerance:
                # Same wounded person!
                competing_drones.append(other_drone_id)
        
        # If multiple drones targeting same person, randomly assign one
        if len(competing_drones) > 1:
            # Deterministic random selection based on drone IDs
            # This ensures all drones agree on who should proceed
            competing_drones.sort()
            chosen_drone = competing_drones[0]  # Lowest ID wins
            
            if chosen_drone != self.identifier:
                print(f"Drone {self.identifier}: Backing off, Drone {chosen_drone} is handling this target.")
                return True  # Back off
        
        return False  # Proceed
    
    def process_lidar_sensor(self) -> bool:
        """
        Returns True if the drone collided with an obstacle.

        Returns:
            bool: True if collision detected, False otherwise.
        """
        # Collision with walls
        if self.lidar_values() is None:
            return False

        dist = min(self.lidar_values())
        return dist < 80

    def process_semantic_sensor_wounded(self):
        """
        Process semantic sensor to find wounded persons.
        
        Returns:
            tuple: (found_free_wounded, command_dict, has_drone_nearby)
                - found_free_wounded: bool indicating if FREE wounded person detected (not grasped)
                - command_dict: movement commands to approach wounded person
                - has_drone_nearby: bool indicating if another drone is near the target
        """
        command = {"forward": 0.5, "lateral": 0.0, "rotation": 0.0}
        
        detection_semantic = self.semantic_values()
        if detection_semantic is None:
            return False, command, False
        
        # Look for wounded persons that are NOT already grasped
        best_angle = 0.0
        best_distance = float('inf')
        found_free_wounded = False
        drone_nearby = False
        
        # First, check if there are other drones near any wounded person we're targeting
        for data in detection_semantic:
            if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                if data.distance < 40:  # Another drone is nearby
                    drone_nearby = True
        
        for data in detection_semantic:
            if (data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON 
                and not data.grasped):  # Only target FREE wounded persons
                found_free_wounded = True
                # Score based on angle and distance (prefer closer and more aligned)
                score = (data.angle ** 2) + (data.distance ** 2 / 10 ** 5)
                if score < (best_angle ** 2) + (best_distance ** 2 / 10 ** 5):
                    best_angle = data.angle
                    best_distance = data.distance
        
        if found_free_wounded:
            # Estimate wounded person's absolute position for communication
            current_pos = self.measured_gps_position()
            angle_abs = self.measured_compass_angle() + best_angle
            self.target_wounded_position = (
                current_pos[0] + best_distance * math.cos(angle_abs),
                current_pos[1] + best_distance * math.sin(angle_abs)
            )
            
            # Simple P controller to turn toward target
            kp = 2.0
            rotation = kp * best_angle
            rotation = max(-1.0, min(1.0, rotation))
            command["rotation"] = rotation
            # Reduce speed if we need to turn a lot
            if abs(rotation) > 0.8:
                command["forward"] = 0.2
        else:
            # No target, clear claim
            self.target_wounded_position = None
        
        return found_free_wounded, command, drone_nearby

    def process_semantic_sensor_rescue_center(self):
        """
        Process semantic sensor to find rescue center.
        
        Returns:
            tuple: (found_rescue_center, command_dict, is_near)
                - found_rescue_center: bool indicating if rescue center detected
                - command_dict: movement commands to approach rescue center
                - is_near: bool indicating if very close to rescue center
        """
        command = {"forward": 0.5, "lateral": 0.0, "rotation": 0.0}
        
        detection_semantic = self.semantic_values()
        if detection_semantic is None:
            return False, command, False
        
        # Look for rescue center
        angles_list = []
        distances_list = []
        found_rescue_center = False
        is_near = False
        
        for data in detection_semantic:
            if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                found_rescue_center = True
                angles_list.append(data.angle)
                distances_list.append(data.distance)
                if data.distance < 30:
                    is_near = True
        
        if found_rescue_center:
            # Calculate mean angle to rescue center
            mean_angle = sum(angles_list) / len(angles_list)
            mean_distance = sum(distances_list) / len(distances_list)
            
            # Save rescue center position if not already saved
            if not self.found_rescue_center:
                self.found_rescue_center = True
                current_pos = self.measured_gps_position()
                # Estimate rescue center position
                angle_abs = self.measured_compass_angle() + mean_angle
                self.rescue_center_position = (
                    current_pos[0] + mean_distance * math.cos(angle_abs),
                    current_pos[1] + mean_distance * math.sin(angle_abs)
                )
                print(f"Drone {self.identifier}: Rescue center located!")
            
            # P controller to turn toward rescue center
            kp = 2.0
            rotation = kp * mean_angle
            rotation = max(-1.0, min(1.0, rotation))
            command["rotation"] = rotation
            
            # Slow down when close
            if is_near:
                command["forward"] = 0.0
                command["rotation"] = -1.0  # Slow rotation to align
            elif abs(rotation) > 0.8:
                command["forward"] = 0.2
        
        return found_rescue_center, command, is_near

    def process_semantic_sensor_col(self):
        '''
        Process semantic sensor to find collision with other drones.
        Returns: (collision_detected, repulsion_command)
        '''
        detection_semantic = self.semantic_values()
        if detection_semantic is None:
            return False, None
        
        closest_drone_distance = float('inf')
        closest_drone_angle = 0.0
        collision = False
        
        for data in detection_semantic:
            if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                if data.distance < 80:
                    collision = True
                if data.distance < closest_drone_distance:
                    closest_drone_distance = data.distance
                    closest_drone_angle = data.angle
        
        # Create repulsion command if drone is too close
        repulsion_command = None
        if collision and closest_drone_distance < 80:
            # Turn away from the closest drone
            repulsion_command = {
                "forward": 0.3,
                "lateral": 0.0,
                "rotation": -1.0 if closest_drone_angle > 0 else 1.0
            }
        
        return collision, repulsion_command
    
    def navigate_to_start(self) -> CommandsDict:
        """
        Navigate back toward the starting position.
        
        Returns:
            CommandsDict: Movement commands to return to start area
        """
        command = {"forward": 0.5, "lateral": 0.0, "rotation": 0.0}
        
        current_pos = self.measured_gps_position()
        if current_pos is None or self.start_position is None:
            # If no GPS, do random search
            return self.control_random_search()
        
        # Calculate direction to start position
        dx = self.start_position[0] - current_pos[0]
        dy = self.start_position[1] - current_pos[1]
        distance_to_start = math.sqrt(dx**2 + dy**2)
        
        # If close enough to start, we're in the return area
        if distance_to_start < 100:
            # Now search for rescue center with semantic sensor
            return {"forward": 0.2, "lateral": 0.0, "rotation": 0.5}
        
        # Calculate target angle
        target_angle = math.atan2(dy, dx)
        current_angle = self.measured_compass_angle()
        if current_angle is None:
            current_angle = 0.0
        
        # P controller for rotation
        diff_angle = normalize_angle(target_angle - current_angle)
        kp = 1.5
        rotation = kp * diff_angle
        rotation = max(-1.0, min(1.0, rotation))
        command["rotation"] = rotation
        
        # Reduce speed if turning
        if abs(rotation) > 0.8:
            command["forward"] = 0.2
        
        return command
    
    def control_random_search(self) -> CommandsDict:
        """
        Random exploration with obstacle avoidance.

        Returns:
            CommandsDict: The control command for random exploration.
        """
        command_straight = {"forward": 1.0, "lateral": 0.0, "rotation": 0.0}
        command_turn = {"forward": 0.0, "lateral": 0.0, "rotation": 1.0}

        drone_collision, repulsion_cmd = self.process_semantic_sensor_col()
        
        # If drone is very close, use repulsion command
        if repulsion_cmd is not None:
            return repulsion_cmd
        
        collided = self.process_lidar_sensor() or drone_collision
        self.counterStraight += 1

        # Start turning if we hit an obstacle or detect a nearby drone
        if collided and not self.isTurning and self.counterStraight > self.distStopStraight:
            self.isTurning = True
            self.angleStopTurning = random.uniform(-math.pi, math.pi)

        # Check if we've finished turning
        measured_angle = self.measured_compass_angle()
        if measured_angle is None:
            measured_angle = 0.0

        diff_angle = normalize_angle(self.angleStopTurning - measured_angle)
        if self.isTurning and abs(diff_angle) < 0.2:
            self.isTurning = False
            self.counterStraight = 0
            self.distStopStraight = random.uniform(10, 50)

        return command_turn if self.isTurning else command_straight

    def control(self) -> CommandsDict:
        """
        Main control loop implementing FSM logic.

        Returns:
            CommandsDict: The control command for the drone.
        """
        command: CommandsDict = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # Save initial position on first call
        if self.start_position is None:
            start_pos = self.measured_gps_position()
            if start_pos is not None:
                self.start_position = (start_pos[0], start_pos[1])
                print(f"Drone {self.identifier}: Start position saved at {self.start_position}")
                # Return base is same as start position
                self.return_base = self.start_position
                print(f"Drone {self.identifier}: Return base set at {self.return_base}")

        # Process communication from other drones
        self.process_communication()

        # Process semantic sensors
        found_free_wounded, wounded_command, drone_near_wounded = self.process_semantic_sensor_wounded()
        found_rescue_center, rescue_command, is_near_rescue = self.process_semantic_sensor_rescue_center()

        # Check if the wounded person we're targeting is already claimed by another drone
        should_back_off = False
        if found_free_wounded and self.target_wounded_position is not None:
            should_back_off = self.is_wounded_claimed(self.target_wounded_position)

        # STATE TRANSITIONS
        if self.state == self.State.SEARCHING_WOUNDED and found_free_wounded and not should_back_off:
            # Only grasp if we found a FREE wounded person and we're not backing off
            self.state = self.State.GRASPING_WOUNDED
            print(f"Drone {self.identifier}: Free Wounded person detected! Switching to GRASPING state.")
        
        elif self.state == self.State.SEARCHING_WOUNDED and found_free_wounded and should_back_off:
            # Another drone is handling this target, continue searching
            self.target_wounded_position = None
            print(f"Drone {self.identifier}: Target already claimed, continuing search.")

        elif self.state == self.State.GRASPING_WOUNDED and self.grasper.grasped_wounded_persons:
            # Successfully grasped - return to base
            self.state = self.State.RETURNING_TO_BASE
            print(f"Drone {self.identifier}: Successfully grasped! Returning to base area.")

        elif self.state == self.State.GRASPING_WOUNDED and should_back_off:
            # Another drone is closer/assigned to this target, back off
            self.state = self.State.SEARCHING_WOUNDED
            self.target_wounded_position = None
            print(f"Drone {self.identifier}: Conflict detected, backing off to search for another target.")

        elif self.state == self.State.GRASPING_WOUNDED and not found_free_wounded:
            # Lost sight of free wounded person (might have been grasped by another drone), go back to searching
            self.state = self.State.SEARCHING_WOUNDED
            self.target_wounded_position = None
            print(f"Drone {self.identifier}: Lost free wounded person, returning to SEARCHING state.")

        elif self.state == self.State.RETURNING_TO_BASE and found_rescue_center and self.grasper.grasped_wounded_persons:
            # Found rescue center, switch to dropping if I have wounded person
            self.state = self.State.DROPPING_RESCUE_CENTER
            print(f"Drone {self.identifier}: Rescue center found! Switching to DROPPING state.")

        elif self.state == self.State.DROPPING_RESCUE_CENTER and not self.grasper.grasped_wounded_persons:
            # Successfully dropped, resume searching
            self.state = self.State.SEARCHING_WOUNDED
            print(f"Drone {self.identifier}: Wounded person dropped! Resuming search.")

        elif self.state == self.State.DROPPING_RESCUE_CENTER and not found_rescue_center:
            # Lost rescue center, go back to returning to base
            self.state = self.State.RETURNING_TO_BASE
            print(f"Drone {self.identifier}: Lost rescue center, returning to base area.")

        # STATE ACTIONS
        if self.state == self.State.SEARCHING_WOUNDED:
            command = self.control_random_search()
            command["grasper"] = 0

        elif self.state == self.State.GRASPING_WOUNDED:
            # Navigate toward wounded person and activate grasper
            command = wounded_command
            command["grasper"] = 1

        elif self.state == self.State.RETURNING_TO_BASE:
            # Navigate back to start area while holding wounded person
            command = self.navigate_to_start()
            command["grasper"] = 1

        elif self.state == self.State.DROPPING_RESCUE_CENTER:
            # Navigate to rescue center and keep grasper on until close
            command = rescue_command
            command["grasper"] = 1

        return command

