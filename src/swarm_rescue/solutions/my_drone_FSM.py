'''
Drone Controller with Finite State Machine (FSM) behavior. This is the main controller. Logic programming based. Yet to add low-level RL policies wherever needed.
'''

import math
import random
from enum import Enum
from typing import Optional, List, Tuple
import numpy as np
from collections import deque
import pickle
from pathlib import Path

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
        RANDOM_EXPLORATION = 5 # Additional state for random exploration if needed
        DONE = 6 # Terminal state 
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

        # For communication
        self.found_rescue_center = False # Used to broadcast rescue center position to other drones
        self.target_wounded_position = None # Communicate which wounded person I'm targeting. To avoid conflicts.
        self.claimed_wounded_positions = {}  # {drone_id: wounded_position}. Avoids conflicts with other drones claiming the same target.
        
        # Random exploration fallback
        self.counterStraight = 0
        self.angleStopTurning = random.uniform(-math.pi, math.pi)
        self.distStopStraight = random.uniform(10, 50)
        self.isTurning = False
        
        # Occupancy grid for frontier-based exploration (LIDAR-only)
        # Grid cell states: 0=UNKNOWN, 1=FREE, 2=OCCUPIED
        self.UNKNOWN = 0
        self.FREE = 1
        self.OCCUPIED = 2
        
        # Initialize grid from map size
        if misc_data and hasattr(misc_data, 'size_area') and misc_data.size_area is not None:
            map_width, map_height = misc_data.size_area
        else:
            # Fallback to reasonable defaults if not available
            map_width, map_height = 1000, 1000
        
        self.grid_resolution = 4  # pixels per cell 
        self.grid_width = int(np.ceil(map_width / self.grid_resolution))
        self.grid_height = int(np.ceil(map_height / self.grid_resolution))
        
        # IMPORTANT: World coordinates are centered at (0,0) with map spanning all 4 quadrants
        # For 1000x1000 map: x goes from -500 to +500, y goes from -500 to +500
        self.grid_origin = (-map_width / 2, -map_height / 2)  
        
        # print(f"[Drone {identifier}] Grid initialized: {self.grid_width}x{self.grid_height} cells")
        # print(f"[Drone {identifier}] World coords: ({self.grid_origin[0]:.0f}, {self.grid_origin[1]:.0f}) to "
            #   f"({self.grid_origin[0] + map_width:.0f}, {self.grid_origin[1] + map_height:.0f})")
        
        # Initialize grid as all UNKNOWN
        self.occupancy_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        
        # Probabilistic occupancy: count hits per cell for temporal filtering
        self.occupied_hits = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        self.free_hits = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        
        # Obstacle inflation for safe navigation (in cells)
        self.obstacle_inflation_radius = 2  # Inflate obstacles by 2 cells (~8px) for safety
        
        # Frontier exploration state
        self.current_frontier_goal = None  # (x, y) in world coordinates
        self.frontier_update_counter = 0
        self.frontier_update_interval = 10  # Update frontiers every N control cycles (increased for stability)
        self.visited_cells = np.zeros((self.grid_height, self.grid_width), dtype=np.uint16)  # Visit frequency
        
        # Stuck detection
        self.stuck_counter = 0
        self.last_position = None
        self.position_history = []  # Track last few positions
        self.min_progress_threshold = 5.0  # Minimum distance to move in 20 cycles
        
        # Sensor range constants
        self.lidar_max_range = 300  # LIDAR max detection range
        self.semantic_max_range = 300  # Semantic sensor max range
        self.sensor_effective_range = 240  # Conservative effective range for frontier filtering
        
        # Visualization export (only drone 0 exports to avoid file conflicts)
        self.enable_viz_export = (identifier == 0)  # Only first drone exports
        self.viz_export_counter = 0
        self.viz_export_interval = 5  # Export every N control cycles
        self.viz_data_file = Path("/tmp/drone_grid_data.pkl")

    def define_message_for_all(self) -> Tuple[Optional[int], Tuple]:  # type: ignore
        """
        Communication between drones controlled by same FSM.
        Message format: (drone_id, (state, rescue_center_pos, target_wounded_pos, current_pos))
        """
        current_pos = self.measured_gps_position()
        msg_data = (
            self.identifier,
            (
                self.state.value,  # Current FSM state
                self.rescue_center_position if self.found_rescue_center else None,
                self.target_wounded_position,  # Which wounded person I'm targeting
                current_pos  # Current position
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
                # print(f"Drone {self.identifier}: Learned rescue center location from Drone {other_drone_id}!")
            
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
                # print(f"Drone {self.identifier}: Backing off, Drone {chosen_drone} is handling this target.")
                return True  # Back off
        
        return False  # Proceed
    
    # ============ OCCUPANCY GRID METHODS ============
    
    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid cell indices.
        Handles center-origin coordinate system where (0,0) is at map center.
        """
        cell_x = int((x - self.grid_origin[0]) / self.grid_resolution)
        cell_y = int((y - self.grid_origin[1]) / self.grid_resolution)
        # Clamp to grid bounds to handle edge cases
        cell_x = max(0, min(cell_x, self.grid_width - 1))
        cell_y = max(0, min(cell_y, self.grid_height - 1))
        return cell_x, cell_y
    
    def cell_to_world(self, cell_x: int, cell_y: int) -> Tuple[float, float]:
        """Convert grid cell indices to world coordinates (cell center)."""
        world_x = self.grid_origin[0] + (cell_x + 0.5) * self.grid_resolution
        world_y = self.grid_origin[1] + (cell_y + 0.5) * self.grid_resolution
        return world_x, world_y
    
    def is_valid_cell(self, cell_x: int, cell_y: int) -> bool:
        """Check if cell indices are within grid bounds."""
        return 0 <= cell_x < self.grid_width and 0 <= cell_y < self.grid_height
    
    def bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm for ray-casting through grid cells."""
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        while True:
            if self.is_valid_cell(x, y):
                cells.append((x, y))
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return cells
    
    def update_occupancy_grid(self):
        """
        Update occupancy grid from LIDAR sensor only.
        Uses temporal filtering to reduce noise and obstacle inflation for safe navigation.
        """
        current_pos = self.measured_gps_position()
        current_angle = self.measured_compass_angle()
        
        if current_pos is None or current_angle is None:
            return # TODO: Implement alternative via odometry if GPS/compass unavailable
        
        drone_cell_x, drone_cell_y = self.world_to_cell(current_pos[0], current_pos[1])
        
        # Mark drone's current cell as visited
        if self.is_valid_cell(drone_cell_x, drone_cell_y):
            self.visited_cells[drone_cell_y, drone_cell_x] += 1
        
        # Update from LIDAR only
        lidar_sensor = self.lidar()
        if lidar_sensor is None:
            return
        
        values = lidar_sensor.get_sensor_values()  # type: ignore
        angles = lidar_sensor.ray_angles  # type: ignore
        max_range = 300  # LIDAR max range
        
        if values is None or angles is None:
            return
        
        # Process every Nth ray for performance (181 rays is overkill)
        ray_skip = 3  # Process every 3rd ray (~60 rays instead of 181)
        
        for i in range(0, len(values), ray_skip):
            measured_range = values[i]
            angle = angles[i]
            
            if measured_range is None or measured_range <= 0:
                continue
            
            # Ray angle in world frame
            ray_angle = current_angle + angle
            
            # Calculate endpoint in world coordinates
            end_x = current_pos[0] + measured_range * math.cos(ray_angle)
            end_y = current_pos[1] + measured_range * math.sin(ray_angle)
            
            end_cell_x, end_cell_y = self.world_to_cell(end_x, end_y)
            
            # Ray-cast from drone to endpoint using Bresenham's algorithm
            cells_on_ray = self.bresenham_line(drone_cell_x, drone_cell_y, end_cell_x, end_cell_y)
            
            # Check if ray hit an obstacle (not just reached max range)
            hit_obstacle = measured_range < 0.999 * max_range
            
            if hit_obstacle:
                # Mark cells along ray as FREE (except last cell)
                for (cx, cy) in cells_on_ray[:-1]:
                    if self.is_valid_cell(cx, cy):
                        self.free_hits[cy, cx] = min(255, self.free_hits[cy, cx] + 1)
                        # Update cell state with temporal filter
                        if self.free_hits[cy, cx] >= 2 and self.occupancy_grid[cy, cx] == self.UNKNOWN:
                            self.occupancy_grid[cy, cx] = self.FREE
                
                # Mark endpoint as OCCUPIED with temporal filter (reduce noise)
                if len(cells_on_ray) > 0:
                    ex, ey = cells_on_ray[-1]
                    if self.is_valid_cell(ex, ey):
                        self.occupied_hits[ey, ex] = min(255, self.occupied_hits[ey, ex] + 1)
                        # Require 3+ hits to mark as OCCUPIED (noise filtering)
                        if self.occupied_hits[ey, ex] >= 3:
                            self.occupancy_grid[ey, ex] = self.OCCUPIED
                            # Inflate obstacle for safety
                            self._inflate_obstacle(ex, ey)
            else:
                # Ray reached max range - mark all cells as FREE (open space)
                for (cx, cy) in cells_on_ray:
                    if self.is_valid_cell(cx, cy):
                        self.free_hits[cy, cx] = min(255, self.free_hits[cy, cx] + 1)
                        if self.free_hits[cy, cx] >= 2 and self.occupancy_grid[cy, cx] == self.UNKNOWN:
                            self.occupancy_grid[cy, cx] = self.FREE
    
    def _inflate_obstacle(self, cell_x: int, cell_y: int):
        """Inflate obstacle by marking nearby cells as occupied for safe navigation."""
        for dy in range(-self.obstacle_inflation_radius, self.obstacle_inflation_radius + 1):
            for dx in range(-self.obstacle_inflation_radius, self.obstacle_inflation_radius + 1):
                # Check if within circular radius
                if dx*dx + dy*dy > self.obstacle_inflation_radius * self.obstacle_inflation_radius:
                    continue
                
                nx, ny = cell_x + dx, cell_y + dy
                if self.is_valid_cell(nx, ny):
                    # Only inflate UNKNOWN or already OCCUPIED cells (don't overwrite FREE)
                    if self.occupancy_grid[ny, nx] in [self.UNKNOWN, self.OCCUPIED]:
                        self.occupancy_grid[ny, nx] = self.OCCUPIED
    
    def detect_frontiers(self) -> List[Tuple[Tuple[float, float], int, int]]:
        """
        Detect frontier cells (free cells adjacent to unknown cells).
        Only returns frontiers that are OUTSIDE current sensor range.
        Returns list of (centroid_world, gain, size) tuples sorted by utility.
        """
        current_pos = self.measured_gps_position()
        if current_pos is None:
            return []
        
        # Find frontier cells: FREE cells with at least one UNKNOWN neighbor
        free_mask = (self.occupancy_grid == self.FREE)
        
        frontier_cells = []
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if not free_mask[y, x]:
                    continue
                
                # Convert to world coordinates to check distance
                world_x, world_y = self.cell_to_world(x, y)
                dx = world_x - current_pos[0]
                dy = world_y - current_pos[1]
                dist_to_cell = math.sqrt(dx*dx + dy*dy)
                
                # SKIP cells within sensor range - already visible!
                if dist_to_cell < self.sensor_effective_range:
                    continue
                
                # Check 8-neighborhood for UNKNOWN cells
                has_unknown_neighbor = False
                occupied_neighbors = 0
                
                for dy_n in [-1, 0, 1]:
                    for dx_n in [-1, 0, 1]:
                        if dx_n == 0 and dy_n == 0:
                            continue
                        ny, nx = y + dy_n, x + dx_n
                        if self.is_valid_cell(nx, ny):
                            if self.occupancy_grid[ny, nx] == self.UNKNOWN:
                                has_unknown_neighbor = True
                            elif self.occupancy_grid[ny, nx] == self.OCCUPIED:
                                occupied_neighbors += 1
                
                # Only add if has unknown neighbor AND not surrounded by walls (dead end)
                if has_unknown_neighbor and occupied_neighbors < 6:
                    frontier_cells.append((x, y))
        
        if not frontier_cells:
            return []
        
        # Group frontier cells into regions using connected components
        frontier_mask = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        for x, y in frontier_cells:
            frontier_mask[y, x] = True
        
        # BFS to find connected components
        visited = np.zeros_like(frontier_mask, dtype=bool)
        regions = []
        
        for x, y in frontier_cells:
            if visited[y, x]:
                continue
            
            # BFS to find all cells in this region
            region = []
            queue = deque([(x, y)])
            visited[y, x] = True
            
            while queue:
                cx, cy = queue.popleft()
                region.append((cx, cy))
                
                # Check 4-neighbors
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if self.is_valid_cell(nx, ny) and frontier_mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))
            
            if len(region) >= 2:  # Only keep regions with at least 2 cells
                regions.append(region)
        
        # Compute centroid, gain, and size for each region
        frontier_info = []
        current_pos = self.measured_gps_position()
        
        if current_pos is None:
            return []
        
        for region in regions:
            # Compute centroid in cell coordinates
            centroid_x = sum(x for x, y in region) / len(region)
            centroid_y = sum(y for x, y in region) / len(region)
            
            # Convert to world coordinates
            centroid_world = self.cell_to_world(int(centroid_x), int(centroid_y))
            
            # CRITICAL: Check if centroid is within sensor range
            dx_cent = centroid_world[0] - current_pos[0]
            dy_cent = centroid_world[1] - current_pos[1]
            dist_to_centroid = math.sqrt(dx_cent*dx_cent + dy_cent*dy_cent)
            
            # Skip frontiers within sensor range
            if dist_to_centroid < self.sensor_effective_range:
                continue
            
            # Information gain: count adjacent unknown cells
            unknown_neighbors = 0
            for x, y in region:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if self.is_valid_cell(nx, ny) and self.occupancy_grid[ny, nx] == self.UNKNOWN:
                            unknown_neighbors += 1
            
            frontier_info.append((centroid_world, unknown_neighbors, len(region)))
        
        # Sort by utility
        def utility(info):
            centroid, gain, size = info
            distance = math.sqrt((centroid[0] - current_pos[0])**2 + (centroid[1] - current_pos[1])**2)
            
            # Check visit frequency at frontier and surrounding area
            cx, cy = self.world_to_cell(centroid[0], centroid[1])
            visit_penalty = 0
            if self.is_valid_cell(cx, cy):
                # Check 5x5 area around frontier centroid
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        check_x, check_y = cx + dx, cy + dy
                        if self.is_valid_cell(check_x, check_y):
                            visit_penalty += self.visited_cells[check_y, check_x]
                
                # MASSIVE penalty for visited areas - we really don't want to go back
                visit_penalty *= 50.0
            
            # Utility: prioritize information gain more, distance less
            # Favor larger frontiers (more exploration potential)
            size_bonus = size * 0.5
            return gain * 3.0 + size_bonus - distance * 0.005 - visit_penalty
        
        frontier_info.sort(key=utility, reverse=True)
        
        return frontier_info
    
    def is_frontier_within_sensor_range(self, frontier_pos: Tuple[float, float]) -> bool:
        """
        Check if a frontier is already within LIDAR/semantic sensor range.
        If we can already see it, no need to navigate there.
        
        Args:
            frontier_pos: (x, y) world coordinates of frontier
            
        Returns:
            bool: True if frontier is within sensor range (don't need to go there)
        """
        current_pos = self.measured_gps_position()
        if current_pos is None:
            return False
        
        # Calculate distance to frontier
        dx = frontier_pos[0] - current_pos[0]
        dy = frontier_pos[1] - current_pos[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        # Use effective sensor range (conservative to account for noise/obstacles)
        return distance < self.sensor_effective_range
    
    def is_path_likely_clear(self, goal: Tuple[float, float]) -> bool:
        """Check if the direct path to goal is likely clear of obstacles."""
        current_pos = self.measured_gps_position()
        if current_pos is None:
            return True
        
        # Sample points along the path
        dx = goal[0] - current_pos[0]
        dy = goal[1] - current_pos[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance < 1:
            return True
        
        # Check every 15 pixels along the path (more dense sampling)
        num_samples = int(distance / 15) + 1
        occupied_count = 0
        unknown_count = 0
        
        for i in range(1, num_samples):
            t = i / num_samples
            sample_x = current_pos[0] + t * dx
            sample_y = current_pos[1] + t * dy
            
            cell_x, cell_y = self.world_to_cell(sample_x, sample_y)
            if self.is_valid_cell(cell_x, cell_y):
                cell_state = self.occupancy_grid[cell_y, cell_x]
                
                # Check 3x3 area around sample point for walls
                for dy_check in [-1, 0, 1]:
                    for dx_check in [-1, 0, 1]:
                        check_x, check_y = cell_x + dx_check, cell_y + dy_check
                        if self.is_valid_cell(check_x, check_y):
                            cell_state = self.occupancy_grid[check_y, check_x]
                            if cell_state == self.OCCUPIED:
                                occupied_count += 1
                            elif cell_state == self.UNKNOWN:
                                unknown_count += 1
        
        # Path is blocked if we detected many walls OR too much unknown (potential walls)
        if occupied_count > 3:  # More than 3 wall cells detected
            return False
        
        # If path goes through mostly unknown territory and we found some walls, be cautious
        if unknown_count > num_samples * 2 and occupied_count > 0:
            return False
        
        return True
    
    def select_frontier_goal(self) -> Optional[Tuple[float, float]]:
        """Select the best frontier and return its world coordinates as goal."""
        frontiers = self.detect_frontiers()
        
        if not frontiers:
            return None
        
        current_pos = self.measured_gps_position()
        if current_pos is None:
            # No GPS, just pick first frontier
            return frontiers[0][0]
        
        # Frontiers are already filtered to be outside sensor range
        # Now just validate paths and select best one
        valid_frontiers = []
        
        for centroid_world, gain, size in frontiers[:10]:  # Check top 10
            distance = math.sqrt((centroid_world[0] - current_pos[0])**2 + 
                               (centroid_world[1] - current_pos[1])**2)
            
            # Check if path is likely clear
            if self.is_path_likely_clear(centroid_world):
                valid_frontiers.append((centroid_world, gain, size, distance))
        
        if not valid_frontiers:
            # All paths seem blocked - try the farthest frontier anyway
            # (maybe our wall detection is overly cautious)
            if frontiers:
                centroid_world, gain, size = frontiers[0]
                distance = math.sqrt((centroid_world[0] - current_pos[0])**2 + 
                                   (centroid_world[1] - current_pos[1])**2)
                return centroid_world
            return None
        
        # Sort by utility: prioritize gain and size, penalize distance
        valid_frontiers.sort(key=lambda f: f[1] * 3.0 + f[2] * 0.5 - f[3] * 0.005, reverse=True)
        
        return valid_frontiers[0][0]
    
    def navigate_to_frontier(self, goal: Tuple[float, float]) -> CommandsDict:
        """
        Navigate toward a frontier goal using P controller with obstacle avoidance.
        
        Args:
            goal: (x, y) world coordinates of target
            
        Returns:
            CommandsDict: Movement commands
        """
        command = {"forward": 0.6, "lateral": 0.0, "rotation": 0.0}
        
        current_pos = self.measured_gps_position()
        current_angle = self.measured_compass_angle()
        
        if current_pos is None or current_angle is None:
            # No GPS/compass, use random exploration as fallback
            return self.control_random_search()
        
        # Calculate direction to goal
        dx = goal[0] - current_pos[0]
        dy = goal[1] - current_pos[1]
        distance_to_goal = math.sqrt(dx**2 + dy**2)
        
        # Check if goal reached - either close enough OR within sensor range
        # No need to go all the way if we can already sense the area
        if distance_to_goal < self.sensor_effective_range:
            # Goal reached - we can now sense this area
            # Immediately try to select a new frontier instead of waiting
            self.current_frontier_goal = None
            self.frontier_update_counter = self.frontier_update_interval  # Force update
            # print(f"Drone {self.identifier}: Reached frontier (within sensor range), selecting new goal")
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.5}
        
        # Get LIDAR readings for obstacle detection
        lidar_values = self.lidar_values()
        obstacle_ahead = False
        
        if lidar_values is not None:
            min_dist = min(lidar_values)
            
            # Check obstacle in forward direction more carefully
            num_rays = len(lidar_values)
            # Check center 60 degrees (30 degrees each side)
            center_start = int(num_rays * 0.4)
            center_end = int(num_rays * 0.6)
            forward_min = min(lidar_values[center_start:center_end]) if center_end > center_start else min_dist
            
            # Proactive obstacle avoidance
            if forward_min < 120:  # Obstacle ahead within 120px (increased)
                obstacle_ahead = True
                
                # If very close (< 70px), this is likely a dead end or wall
                # Abandon goal and mark this area as problematic
                if min_dist < 70:
                    if self.current_frontier_goal is not None:
                        # Mark cells near goal as occupied to avoid retrying
                        goal_cell_x, goal_cell_y = self.world_to_cell(
                            self.current_frontier_goal[0], 
                            self.current_frontier_goal[1]
                        )
                        # Mark larger area (7x7) to really avoid this dead end
                        for dy in range(-3, 4):
                            for dx in range(-3, 4):
                                gx, gy = goal_cell_x + dx, goal_cell_y + dy
                                if self.is_valid_cell(gx, gy):
                                    self.occupancy_grid[gy, gx] = self.OCCUPIED
                                    # Also mark as heavily visited
                                    self.visited_cells[gy, gx] += 10
                    
                    self.current_frontier_goal = None
                    self.stuck_counter += 1
                    self.frontier_update_counter = self.frontier_update_interval  # Force new goal
                    # print(f"Drone {self.identifier}: Hit wall, marking dead end and finding new goal")
                    
                    # Turn away from obstacle
                    mid = num_rays // 2
                    left_min = min(lidar_values[:mid]) if mid > 0 else 300
                    right_min = min(lidar_values[mid:]) if mid < num_rays else 300
                    
                    if left_min > right_min:
                        command["rotation"] = 1.0
                    else:
                        command["rotation"] = -1.0
                    command["forward"] = 0.1
                    return command
        
        # Calculate target angle
        target_angle = math.atan2(dy, dx)
        diff_angle = normalize_angle(target_angle - current_angle)
        
        # P controller for rotation
        kp_rotation = 1.2
        rotation = kp_rotation * diff_angle
        rotation = max(-1.0, min(1.0, rotation))
        command["rotation"] = float(rotation)
        
        # Adjust speed based on obstacles and turning
        if obstacle_ahead:
            # Slow down and turn more when obstacle ahead
            command["forward"] = 0.2
            command["rotation"] *= 1.5  # Turn more aggressively
            command["rotation"] = max(-1.0, min(1.0, command["rotation"]))
        elif abs(rotation) > 0.8:
            command["forward"] = 0.3
        elif abs(rotation) > 0.5:
            command["forward"] = 0.5
        else:
            command["forward"] = 0.7
        
        return command
    
    # ============ END OCCUPANCY GRID METHODS ============
    
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
        return dist < 60  # Reduced threshold for better safety margin

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
        
        # First, check if there are other drones near any wounded person we're targeting - to handle race conditions
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
            current_angle = self.measured_compass_angle()
            if current_pos is not None and current_angle is not None:
                angle_abs = current_angle + best_angle
                self.target_wounded_position = (
                    current_pos[0] + best_distance * math.cos(angle_abs),
                    current_pos[1] + best_distance * math.sin(angle_abs)
                )
            else:
                self.target_wounded_position = None
            
            # Simple P controller to turn toward target
            kp = 2.0
            rotation = kp * best_angle
            rotation = max(-1.0, min(1.0, rotation))
            command["rotation"] = float(rotation)
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
                current_pos = self.measured_gps_position()
                current_angle = self.measured_compass_angle()
                if current_pos is not None and current_angle is not None:
                    self.found_rescue_center = True
                    # Estimate rescue center position
                    angle_abs = current_angle + mean_angle
                    self.rescue_center_position = (
                        current_pos[0] + mean_distance * math.cos(angle_abs),
                        current_pos[1] + mean_distance * math.sin(angle_abs)
                    )
                    # print(f"Drone {self.identifier}: Rescue center located!")
            
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
        command["rotation"] = float(rotation)
        
        # Reduce speed if turning
        if abs(rotation) > 0.8:
            command["forward"] = 0.2
        
        return command
    
    def export_visualization_data(self):
        """Export grid data for real-time visualization in separate window"""
        if not self.enable_viz_export:
            return
        
        try:
            # Gather current frontiers
            frontiers = self.detect_frontiers()
            
            # Gather all drone positions from communication
            drone_positions = []
            current_pos = self.measured_gps_position()
            if current_pos is not None:
                drone_positions.append((current_pos[0], current_pos[1]))
            
            # Add other drones' positions from messages
            if self.communicator and self.communicator.received_messages:
                for msg in self.communicator.received_messages:
                    message = msg[1]
                    other_position = message[1][3]  # (state, rescue_pos, target, position)
                    if other_position is not None:
                        drone_positions.append((other_position[0], other_position[1]))
            
            data = {
                'occupancy_grid': self.occupancy_grid.copy(),
                'visited_cells': self.visited_cells.copy(),
                'frontiers': frontiers[:20],  # Top 20 frontiers only
                'current_goal': self.current_frontier_goal,
                'drone_positions': drone_positions,
                'grid_origin': self.grid_origin,
                'grid_resolution': self.grid_resolution,
                'timestamp': self.viz_export_counter
            }
            
            # Write atomically (write to temp file, then rename)
            temp_file = self.viz_data_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_file.replace(self.viz_data_file)
            
        except Exception as e:
            # Don't crash if visualization export fails
            pass
    
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

        # ON FIRST CALL

        # Save initial position and return base for later use
        if self.start_position is None:
            start_pos = self.measured_gps_position()
            if start_pos is not None:
                self.start_position = (start_pos[0], start_pos[1])
                # print(f"Drone {self.identifier}: Start position saved at {self.start_position}")
                # Return base is same as start position
                self.return_base = self.start_position
                # print(f"Drone {self.identifier}: Return base set at {self.return_base}")
        
        # TODO: Else use odometer if no GPS available to find start position

        # ----

        # Update occupancy grid from sensors
        self.update_occupancy_grid()
        
        # Export visualization data (only drone 0, periodically)
        if self.enable_viz_export:
            self.viz_export_counter += 1
            if self.viz_export_counter % self.viz_export_interval == 0:
                self.export_visualization_data()
        
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
            # print(f"Drone {self.identifier}: Free Wounded person detected! Switching to GRASPING state.")
        
        elif self.state == self.State.SEARCHING_WOUNDED and found_free_wounded and should_back_off:
            # Another drone is handling this target, continue searching
            self.target_wounded_position = None
            # print(f"Drone {self.identifier}: Target already claimed, continuing search.")

        elif self.state == self.State.GRASPING_WOUNDED and self.grasper.grasped_wounded_persons:
            # Successfully grasped - return to base
            self.state = self.State.RETURNING_TO_BASE
            # print(f"Drone {self.identifier}: Successfully grasped! Returning to base area.")

        elif self.state == self.State.GRASPING_WOUNDED and should_back_off:
            # Another drone is closer/assigned to this target, back off
            self.state = self.State.SEARCHING_WOUNDED
            self.target_wounded_position = None
            # print(f"Drone {self.identifier}: Conflict detected, backing off to search for another target.")

        elif self.state == self.State.GRASPING_WOUNDED and not found_free_wounded:
            # Lost sight of free wounded person (might have been grasped by another drone), go back to searching
            self.state = self.State.SEARCHING_WOUNDED
            self.target_wounded_position = None
            # print(f"Drone {self.identifier}: Lost free wounded person, returning to SEARCHING state.")

        elif self.state == self.State.RETURNING_TO_BASE and found_rescue_center and self.grasper.grasped_wounded_persons:
            # Found rescue center, switch to dropping if I have wounded person
            self.state = self.State.DROPPING_RESCUE_CENTER
            # print(f"Drone {self.identifier}: Rescue center found! Switching to DROPPING state.")

        elif self.state == self.State.DROPPING_RESCUE_CENTER and not self.grasper.grasped_wounded_persons:
            # Successfully dropped, resume searching
            self.state = self.State.SEARCHING_WOUNDED
            # print(f"Drone {self.identifier}: Wounded person dropped! Resuming search.")

        elif self.state == self.State.DROPPING_RESCUE_CENTER and not found_rescue_center:
            # Lost rescue center, go back to returning to base
            self.state = self.State.RETURNING_TO_BASE
            # print(f"Drone {self.identifier}: Lost rescue center, returning to base area.")

        # STATE ACTIONS
        if self.state == self.State.SEARCHING_WOUNDED:
            # Frontier-based exploration with occupancy grid
            
            # Track position for stuck detection
            current_pos = self.measured_gps_position()
            if current_pos is not None:
                self.position_history.append(current_pos)
                if len(self.position_history) > 20:
                    self.position_history.pop(0)
                
                # Check if stuck (not moving much in last 20 cycles)
                if len(self.position_history) >= 20:
                    first_pos = self.position_history[0]
                    last_pos = self.position_history[-1]
                    progress = math.sqrt((last_pos[0] - first_pos[0])**2 + 
                                       (last_pos[1] - first_pos[1])**2)
                    
                    if progress < self.min_progress_threshold:
                        # Stuck! Clear goal and reset
                        self.current_frontier_goal = None
                        self.stuck_counter += 1
                        # print(f"Drone {self.identifier}: Stuck detected, clearing goal. Stuck count: {self.stuck_counter}")
            
            # Update frontier goal periodically or if stuck or if no goal
            self.frontier_update_counter += 1
            should_update = (self.frontier_update_counter >= self.frontier_update_interval or 
                           self.stuck_counter > 2 or  # Reduced from 3
                           self.current_frontier_goal is None)
            
            if should_update:
                self.frontier_update_counter = 0
                
                # Reset stuck counter when we try new goal
                if self.stuck_counter > 2:
                    self.stuck_counter = 0
                    self.position_history.clear()
                    # print(f"Drone {self.identifier}: Stuck detected, resetting")
                
                # Try to find a new frontier
                new_goal = self.select_frontier_goal()
                if new_goal is not None:
                    old_goal = self.current_frontier_goal
                    self.current_frontier_goal = new_goal
                    if old_goal != new_goal:
                        curr_pos = self.measured_gps_position()
                        if curr_pos is not None:
                            dist = math.sqrt((new_goal[0]-curr_pos[0])**2 + (new_goal[1]-curr_pos[1])**2)
                            # print(f"Drone {self.identifier}: New frontier at ({new_goal[0]:.0f}, {new_goal[1]:.0f}), dist={dist:.0f}px")
                else:
                    # print(f"Drone {self.identifier}: No valid frontiers found (all sensed/visited/blocked)")
                    pass
            
            # Navigate to current frontier goal if available
            if self.current_frontier_goal is not None:
                command = self.navigate_to_frontier(self.current_frontier_goal)
            else:
                # No frontiers available, use random exploration as fallback
                command = self.control_random_search()
                self.stuck_counter = max(0, self.stuck_counter - 1)  # Reduce stuck counter during random
            
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

