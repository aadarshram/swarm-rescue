import pathlib
import sys
from typing import Type

import cv2
import numpy as np


# Insert the 'src' directory, located two levels up from the current script,
# into sys.path. This ensures Python can find project-specific modules
# (e.g., 'swarm_rescue') when the script is run from a subfolder like 'examples/'.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.maps.walls_medium_02 import add_walls, add_boxes
from swarm_rescue.simulation.utils.constants import MAX_RANGE_LIDAR_SENSOR
from swarm_rescue.simulation.utils.grid import Grid
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.utils.pose import Pose
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.elements.rescue_center import RescueCenter
from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
from swarm_rescue.simulation.gui_map.map_abstract import MapAbstract
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor


class OccupancyGrid(Grid):
    """Simple occupancy grid"""

    def __init__(self,
                 size_area_world,
                 resolution: float,
                 lidar, semantic):
        super().__init__(size_area_world=size_area_world,
                         resolution=resolution)

        self.size_area_world = size_area_world
        self.resolution = resolution

        self.lidar = lidar
        self.semantic = semantic

        self.x_max_grid: int = int(self.size_area_world[0] / self.resolution
                                   + 0.5)
        self.y_max_grid: int = int(self.size_area_world[1] / self.resolution
                                   + 0.5)

        self.grid = np.zeros((self.x_max_grid, self.y_max_grid))
        self.zoomed_grid = np.empty((self.x_max_grid, self.y_max_grid))

    def update_grid(self, pose: Pose):
        """
        Bayesian map update with new observation
        lidar : lidar data
        pose : corrected pose in world coordinates
        """
        EVERY_N = 3
        LIDAR_DIST_CLIP = 40.0
        EMPTY_ZONE_VALUE = -0.602
        OBSTACLE_ZONE_VALUE = 2.0
        FREE_ZONE_VALUE = -4.0
        THRESHOLD_MIN = -40
        THRESHOLD_MAX = 40
        PERSON_VALUE = 100
        RESCUE_CENTER_VALUE = 60

        lidar_dist = self.lidar.get_sensor_values()[::EVERY_N].copy()
        lidar_angles = self.lidar.ray_angles[::EVERY_N].copy()


        # Compute cos and sin of the absolute angle of the lidar
        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)

        max_range = MAX_RANGE_LIDAR_SENSOR * 0.9

        # For empty zones
        # points_x and point_y contains the border of detected empty zone
        # We use a value a little bit less than LIDAR_DIST_CLIP because of the
        # noise in lidar
        lidar_dist_empty = np.maximum(lidar_dist - LIDAR_DIST_CLIP, 0.0)
        # All values of lidar_dist_empty_clip are now <= max_range
        lidar_dist_empty_clip = np.minimum(lidar_dist_empty, max_range)
        points_x = pose.position[0] + np.multiply(lidar_dist_empty_clip,
                                                  cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist_empty_clip,
                                                  sin_rays)

        for pt_x, pt_y in zip(points_x, points_y):
            self.add_value_along_line(pose.position[0], pose.position[1],
                                      pt_x, pt_y,
                                      EMPTY_ZONE_VALUE)

        # For obstacle zones, all values of lidar_dist are < max_range
        select_collision = lidar_dist < max_range

        points_x = pose.position[0] + np.multiply(lidar_dist, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist, sin_rays)

        points_x = points_x[select_collision]
        points_y = points_y[select_collision]

        self.add_points(points_x, points_y, OBSTACLE_ZONE_VALUE)

        # the current position of the drone is free !
        self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)

        # threshold values
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)

        semantic_values = self.semantic.get_sensor_values()
        for data in semantic_values:
            if (data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped):
                d = data.distance
                a = data.angle + pose.orientation
                x = pose.position[0] + d*np.cos(a)
                y = pose.position[1] + d*np.sin(a)
                self.add_points(x, y, PERSON_VALUE)
            if (data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER):
                d = data.distance
                a = data.angle + pose.orientation
                x = pose.position[0] + d*np.cos(a)
                y = pose.position[1] + d*np.sin(a)
                self.add_points(x, y, RESCUE_CENTER_VALUE)



        # compute zoomed grid for displaying
        self.zoomed_grid = self.grid.copy()
        new_zoomed_size = (int(self.size_area_world[1] * 0.5),
                           int(self.size_area_world[0] * 0.5))
        self.zoomed_grid = cv2.resize(self.zoomed_grid, new_zoomed_size,
                                      interpolation=cv2.INTER_NEAREST)


def find_pose(drone:DroneAbstract, pose = Pose()):
    gps_sensor = drone.gps_values()
    compass_sensor = drone.compass_values()
    odometer = drone.odometer_values()

    # Finding Pose, assuming GPS gives absolute position. In no gps zones, we use odom.
    if gps_sensor is not None and compass_sensor is not None: 
        pose = Pose(np.asarray(gps_sensor.position),compass_sensor.angle)
    elif odometer is not None:
        dist_travel = getattr(odometer, "delta_distance", 0.0)
        alpha = getattr(odometer, "delta_alpha", 0.0)
        theta = getattr(odometer, "delta_theta", 0.0)

        beta = pose.orientation + alpha
        x = pose.position[0] + dist_travel*np.cos(beta)
        y = pose.position[1] + dist_travel*np.sin(beta)
        angle = pose.orientation + theta
        pose = Pose(np.asarray((x,y)),angle)
    return pose