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
from scipy import interpolate

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
        
        # Step counter for timer-based logic
        self.step_count = 0
        
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
        
        self.grid_resolution = 15
        self.grid_width = int(np.ceil(map_width / self.grid_resolution))
        self.grid_height = int(np.ceil(map_height / self.grid_resolution))
        
        # IMPORTANT: World coordinates are centered at (0,0) with map spanning all 4 quadrants
        # For 1000x1000 map: x goes from -500 to +500, y goes from -500 to +500
        self.grid_origin = (-map_width / 2, -map_height / 2)  
    
        # Initialize grid as all UNKNOWN
        self.occupancy_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        self.inflation_radius = 2  # Inflate obstacles by 2 cells (30px safety margin)
        self.inflated_grid = None  # Inflated version for safe path planning
        
        # Global occupancy grid (aggregated from all drones) - only maintained by drone 0 for visualization
        self.global_occupancy_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8) if identifier == 0 else None
        
        # Frontier exploration state
        self.current_frontier_goal = None  # (x, y) in world coordinates
        self.current_path = None  # List of (x, y) world coordinates along planned path
        self.path_waypoint_index = 0  # Current waypoint along path
        self.goal_start_time = 0  # Step count when current goal was set
        self.goal_timeout_threshold = 300  # Reset goal after 300 steps (~10 sec at 30Hz)
        
        # PID controller state
        self.last_angle_error = 0.0  # For derivative term
        self.angle_error_integral = 0.0  # For integral term
   
        self.visited_cells = np.zeros((self.grid_height, self.grid_width), dtype=np.uint16)  # Visit frequency
        
        # Grid sharing for multi-drone coordination
        self.grid_changed_cells = set()  # Track cells updated since last communication
        self.last_shared_timestamp = 0
        
        # Sensor range constants
        self.lidar_max_range = 300  # LIDAR max detection range
        self.semantic_max_range = 300  # Semantic sensor max range
        self.sensor_effective_range = 240  # Conservative effective range for frontier filtering
        

        # DEBUG flags
        self.gps()._noise = False
        self.compass()._noise = False
        self.odometer()._noise = False
        self.lidar()._noise = False  # type: ignore
        # Visualization export (only drone 0 exports to avoid file conflicts)
        self.enable_viz_export = False # (identifier == 0)  # Only first drone exports
        self.viz_export_counter = 0
        self.viz_export_interval = 5  # Export every N control cycles
        self.viz_data_file = Path("/tmp/drone_grid_data.pkl")

    def define_message_for_all(self) -> Tuple[Optional[int], Tuple]:  # type: ignore
        """
        Communication between drones controlled by same FSM.
        Message format: (drone_id, (state, rescue_center_pos, target_wounded_pos, current_pos, grid_updates))
        """
        current_pos = self.measured_gps_position()
        
        # Package grid updates (only send changed cells to reduce data)
        grid_updates = None
        if len(self.grid_changed_cells) > 0:
            # Send list of (x, y, value) tuples for changed cells
            grid_updates = [(x, y, int(self.occupancy_grid[y, x])) 
                           for x, y in list(self.grid_changed_cells)[:100]]  # Limit to 100 cells per message
            self.grid_changed_cells.clear()
        
        msg_data = (
            self.identifier,
            (
                self.state.value,  # Current FSM state
                self.rescue_center_position if self.found_rescue_center else None,
                self.target_wounded_position,  # Which wounded person I'm targeting
                current_pos,  # Current position
                grid_updates  # Occupancy grid updates
            )
        )
        return msg_data
    
    def process_communication(self):
        """
        Process messages from other drones to:
        1. Learn rescue center location from others
        2. Know which wounded persons are claimed by other drones
        3. Track other drones' positions to avoid clustering
        4. Merge occupancy grid updates from other drones
        """
        if not self.communicator:
            return
        
        received_messages = self.communicator.received_messages
        
        # Clear old data
        self.claimed_wounded_positions.clear()
        
        for msg in received_messages:
            message = msg[1]
            other_drone_id = message[0]
            other_state, other_rescue_pos, other_target_wounded, other_position, grid_updates = message[1]
            
            # Learn rescue center location from other drones
            if other_rescue_pos is not None and not self.found_rescue_center:
                self.rescue_center_position = other_rescue_pos
                self.found_rescue_center = True
                # print(f"Drone {self.identifier}: Learned rescue center location from Drone {other_drone_id}!")
            
            # Track which wounded persons are claimed by other drones
            if other_target_wounded is not None:
                self.claimed_wounded_positions[other_drone_id] = other_target_wounded
            
            # Merge occupancy grid updates from other drone
            if grid_updates is not None:
                for x, y, value in grid_updates:
                    if self.is_valid_cell(x, y):
                        # Merge into local grid
                        # Merge strategy: OCCUPIED takes priority (union of all obstacles)
                        if value == self.OCCUPIED:
                            self.occupancy_grid[y, x] = self.OCCUPIED
                        elif value == self.FREE and self.occupancy_grid[y, x] == self.UNKNOWN:
                            # Only update UNKNOWN to FREE, don't overwrite OCCUPIED
                            self.occupancy_grid[y, x] = self.FREE
                        
                        # Also merge into global grid (only for drone 0)
                        if self.global_occupancy_grid is not None:
                            if value == self.OCCUPIED:
                                self.global_occupancy_grid[y, x] = self.OCCUPIED
                            elif value == self.FREE and self.global_occupancy_grid[y, x] == self.UNKNOWN:
                                self.global_occupancy_grid[y, x] = self.FREE

    
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
    
    # OCCUPANCY GRID METHODS 
    
    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to grid cell indices.
        """
        cell_x = math.floor((x - self.grid_origin[0]) / self.grid_resolution)
        cell_y = math.floor((y - self.grid_origin[1]) / self.grid_resolution)

        # Clamp to grid bounds to handle edge cases
        cell_x = max(0, min(cell_x, self.grid_width - 1))
        cell_y = max(0, min(cell_y, self.grid_height - 1))
        return cell_x, cell_y
    
    def cell_to_world(self, cell_x: int, cell_y: int) -> Tuple[float, float]:
        """
        Convert grid cell indices to world coordinates.
        """
        world_x = self.grid_origin[0] + (cell_x + 0.5) * self.grid_resolution
        world_y = self.grid_origin[1] + (cell_y + 0.5) * self.grid_resolution
        return world_x, world_y
    
    def is_valid_cell(self, cell_x: int, cell_y: int) -> bool:
        """Check if cell indices are within grid bounds."""
        return 0 <= cell_x < self.grid_width and 0 <= cell_y < self.grid_height
    
    def bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """
        Bresenham's line algorithm for ray-casting through grid cells.
        """
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
        """

        # Update current position

        current_pos = self.measured_gps_position()
        current_angle = self.measured_compass_angle()
        if current_pos is None or current_angle is None:
            return # TODO: Implement alternative via odometry if GPS/compass unavailable
        
        drone_cell_x, drone_cell_y = self.world_to_cell(current_pos[0], current_pos[1])
        
        if self.is_valid_cell(drone_cell_x, drone_cell_y):
            self.visited_cells[drone_cell_y, drone_cell_x] += 1
        
        # STEP 1: Update from LIDAR - mark all obstacles as OCCUPIED (standard ray-casting)
        lidar_sensor = self.lidar()
        if lidar_sensor is None:
            return # fallback
        
        values = lidar_sensor.get_sensor_values()  # type: ignore
        angles = lidar_sensor.ray_angles  # type: ignore
        max_range = self.lidar_max_range
        
        if values is None or angles is None:
            return # fallback
        
        # Process every Nth ray
        ray_skip = 1
        for i in range(0, len(values), ray_skip):
            measured_range = values[i]
            angle = angles[i]  # Relative angle from drone's perspective
            # Ray angle in world frame
            ray_angle = current_angle + angle
            # Calculate endpoint this is either an obstacle or max range
            end_x = current_pos[0] + measured_range * math.cos(ray_angle)
            end_y = current_pos[1] + measured_range * math.sin(ray_angle)
            
            end_cell_x, end_cell_y = self.world_to_cell(end_x, end_y)
            
            # Ray-cast from drone to endpoint using Bresenham's algorithm
            cells_on_ray = self.bresenham_line(drone_cell_x, drone_cell_y, end_cell_x, end_cell_y)
            
            # Check if ray hit an obstacle
            hit_obstacle = measured_range < 0.999 * max_range
            
            if hit_obstacle:
                # Mark cells along ray as FREE (except last cell)
                for (cx, cy) in cells_on_ray[:-1]:
                    if self.is_valid_cell(cx, cy):
                        if self.occupancy_grid[cy, cx] != self.OCCUPIED:  # Persist occupied cells
                            old_value = self.occupancy_grid[cy, cx]
                            self.occupancy_grid[cy, cx] = self.FREE
                            if old_value != self.FREE:
                                self.grid_changed_cells.add((cx, cy))
                
                # Mark endpoint as OCCUPIED (all obstacles - walls, drones, wounded, etc.)
                if len(cells_on_ray) > 0:
                    ex, ey = cells_on_ray[-1]
                    if self.is_valid_cell(ex, ey):
                        old_value = self.occupancy_grid[ey, ex]
                        self.occupancy_grid[ey, ex] = self.OCCUPIED
                        if old_value != self.OCCUPIED:
                            self.grid_changed_cells.add((ex, ey))
            else:
                # Ray reached max range - mark all cells as FREE (open space)
                for (cx, cy) in cells_on_ray:
                    if self.is_valid_cell(cx, cy):
                        if self.occupancy_grid[cy, cx] != self.OCCUPIED:  # Persist occupied cells
                            old_value = self.occupancy_grid[cy, cx]
                            self.occupancy_grid[cy, cx] = self.FREE
                            if old_value != self.FREE:
                                self.grid_changed_cells.add((cx, cy))
        
        # STEP 2: Override with semantic sensor - clear non-wall obstacles
        semantic_data = self.semantic_values()
        if semantic_data is not None:
            for data in semantic_data:
                # Semantic detects: WOUNDED_PERSON, DRONE, RESCUE_CENTER (NOT walls)
                # Calculate position of detected entity in world coordinates
                detection_angle = current_angle + data.angle  # World frame angle
                detection_x = current_pos[0] + data.distance * math.cos(detection_angle)
                detection_y = current_pos[1] + data.distance * math.sin(detection_angle)
                
                # Convert to grid cell
                cell_x, cell_y = self.world_to_cell(detection_x, detection_y)
                
                # Clear OCCUPIED marking for non-wall entities (with buffer to handle uncertainty)
                for dy in range(-1, 2):  # 3x3 cell area around detection
                    for dx in range(-1, 2):
                        cx, cy = cell_x + dx, cell_y + dy
                        if self.is_valid_cell(cx, cy):
                            # Override: mark as FREE since semantic identified it as non-wall
                            old_value = self.occupancy_grid[cy, cx]
                            self.occupancy_grid[cy, cx] = self.FREE
                            if old_value != self.FREE:
                                self.grid_changed_cells.add((cx, cy))
        
        # TODO: Improve for probabilistic updates later when noise is enabled
    
    def inflate_obstacles(self) -> np.ndarray:
        """Create inflated version of occupancy grid with safety buffer around obstacles.
        
        Returns:
            Inflated grid where obstacles are expanded by inflation_radius cells
        """
        inflated = self.occupancy_grid.copy()
        
        # Find all occupied cells
        occupied_cells = np.argwhere(self.occupancy_grid == self.OCCUPIED)
        
        # Inflate each occupied cell
        for y, x in occupied_cells:
            # Inflate in square around obstacle
            for dy in range(-self.inflation_radius, self.inflation_radius + 1):
                for dx in range(-self.inflation_radius, self.inflation_radius + 1):
                    ny, nx = y + dy, x + dx
                    if self.is_valid_cell(nx, ny):
                        # Only inflate into FREE or UNKNOWN cells, not other obstacles
                        if inflated[ny, nx] != self.OCCUPIED:
                            inflated[ny, nx] = self.OCCUPIED
        
        return inflated
    
    def detect_frontiers(self) -> List[Tuple[Tuple[float, float], int, int]]:
        """
        Detect frontier cells (free cells adjacent to unknown cells).
        Only returns frontiers that are OUTSIDE current sensor range.
        Returns list of (centroid_world, gain, size) tuples sorted by utility.
        """
        current_pos = self.measured_gps_position()
        if current_pos is None:
            return []
        
        # Use inflated grid to avoid frontiers near obstacles
        inflated_grid = self.inflate_obstacles()
        
        # Find frontier cells: FREE cells with at least one UNKNOWN neighbor
        # Use inflated grid to exclude cells too close to obstacles
        free_mask = (self.occupancy_grid == self.FREE) & (inflated_grid == self.FREE)
        
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
                        nx, ny = x + dx_n, y + dy_n
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
                        nx, ny = x + dx, y + dy
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
    
    def is_frontier_reachable_bfs(self, goal: Tuple[float, float], max_search_cells: int = 2000) -> Optional[List[Tuple[int, int]]]:
        """Use BFS to check if there's a path from drone to frontier. Returns path as list of (x,y) cell coords, or None if unreachable."""
        current_pos = self.measured_gps_position()
        if current_pos is None:
            return None
        
        # Convert positions to grid cells
        start_x, start_y = self.world_to_cell(current_pos[0], current_pos[1])
        goal_x, goal_y = self.world_to_cell(goal[0], goal[1])
        
        if not self.is_valid_cell(start_x, start_y) or not self.is_valid_cell(goal_x, goal_y):
            return None
        
        # Use inflated grid for safer path planning
        planning_grid = self.inflate_obstacles()
        
        # BFS to find path - track parent for path reconstruction
        visited = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        parent = {}  # Store parent cell for path reconstruction
        queue = deque([(start_x, start_y)])
        visited[start_y, start_x] = True
        parent[(start_x, start_y)] = None
        cells_searched = 0
        
        goal_reached = False
        goal_cell = None
        
        while queue and cells_searched < max_search_cells:
            cx, cy = queue.popleft()
            cells_searched += 1
            
            # Check if we reached the goal
            if abs(cx - goal_x) <= 2 and abs(cy - goal_y) <= 2:  # Within 2 cells is close enough
                goal_reached = True
                goal_cell = (cx, cy)
                break
            
            # Explore 8-connected neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx, ny = cx + dx, cy + dy
                    
                    if not self.is_valid_cell(nx, ny) or visited[ny, nx]:
                        continue
                    
                    # Use inflated grid - only traverse FREE or UNKNOWN cells
                    # This keeps paths away from obstacles
                    cell_state = planning_grid[ny, nx]
                    if cell_state != self.OCCUPIED:
                        visited[ny, nx] = True
                        parent[(nx, ny)] = (cx, cy)
                        queue.append((nx, ny))
        
        if not goal_reached:
            return None  # Path not found
        
        # Reconstruct path from goal to start
        path = []
        current = goal_cell
        while current is not None:
            path.append(current)
            current = parent.get(current)
        
        path.reverse()  # Reverse to get start-to-goal path
        return path
    
    def smooth_path_bspline(self, path_cells: List[Tuple[int, int]], num_points: int = 50) -> List[Tuple[float, float]]:
        """Smooth the BFS path using B-spline interpolation.
        
        Args:
            path_cells: List of (x, y) cell coordinates from BFS
            num_points: Number of interpolated points to generate
            
        Returns:
            List of (x, y) world coordinates along smoothed path
        """
        if path_cells is None or len(path_cells) < 3:
            # Not enough points for spline, return original path in world coords
            if path_cells is not None:
                return [self.cell_to_world(x, y) for x, y in path_cells]
            return []
        
        # Convert cell coordinates to world coordinates
        world_coords = np.array([self.cell_to_world(x, y) for x, y in path_cells])
        x_coords = world_coords[:, 0]
        y_coords = world_coords[:, 1]
        
        # Remove duplicate consecutive points
        unique_indices = [0]
        for i in range(1, len(x_coords)):
            if x_coords[i] != x_coords[unique_indices[-1]] or y_coords[i] != y_coords[unique_indices[-1]]:
                unique_indices.append(i)
        
        x_coords = x_coords[unique_indices]
        y_coords = y_coords[unique_indices]
        
        if len(x_coords) < 3:
            # Still not enough unique points
            return [(x_coords[i], y_coords[i]) for i in range(len(x_coords))]
        
        # Parametric B-spline interpolation
        # Degree k: use min of 3 or (num_points - 1) to avoid errors
        k = min(3, len(x_coords) - 1)
        
        try:
            # Create parametric spline
            tck, u = interpolate.splprep([x_coords, y_coords], s=0, k=k)
            
            # Generate smooth path with specified number of points
            u_new = np.linspace(0, 1, num_points)
            smooth_coords = interpolate.splev(u_new, tck)
            
            # Convert to list of tuples - smooth_coords is tuple of arrays
            x_smooth = np.array(smooth_coords[0])  # type: ignore
            y_smooth = np.array(smooth_coords[1])  # type: ignore
            smooth_path = [(float(x_smooth[i]), float(y_smooth[i])) 
                          for i in range(len(x_smooth))]
            return smooth_path
            
        except Exception as e:
            # If spline fails, return original path
            return [(x_coords[i], y_coords[i]) for i in range(len(x_coords))]
    
    def select_frontier_goal(self) -> Optional[Tuple[float, float]]:
        """
        Select the best frontier and return its world coordinates as goal.
        Uses BFS to verify path reachability through free cells.
        """

        # Detect all frontiers
        frontiers = self.detect_frontiers()
        if not frontiers:
            return None
        
        # Return the best frontier based on objective
        current_pos = self.measured_gps_position()
        if current_pos is None:
            return None # TODO: Implement alternative via odometry if GPS unavailable

        # Filter frontiers by BFS reachability and store paths
        reachable_frontiers = []
        frontier_paths = {}  # Map centroid to path
        
        for centroid, gain, size in frontiers[:15]:  # Check top 15 to save computation
            path = self.is_frontier_reachable_bfs(centroid)
            if path is not None:
                reachable_frontiers.append((centroid, gain, size))
                frontier_paths[centroid] = path
        
        # If no reachable frontiers found, fall back to closest ones (maybe grid is incomplete)
        if not reachable_frontiers:
            reachable_frontiers = frontiers[:5]

        def utility(f):
            centroid, gain, size = f
            dx = centroid[0] - current_pos[0]
            dy = centroid[1] - current_pos[1]
            distance = math.hypot(dx, dy)
            return gain * 3.0 + size * 0.5 - distance * 0.005 # Objective 

        reachable_frontiers.sort(key=utility, reverse=True)
        best_frontier = reachable_frontiers[0][0]
        
        # Store the path for this frontier if available
        if best_frontier in frontier_paths:
            raw_path = frontier_paths[best_frontier]
            # Smooth the path using B-splines - fewer points for cleaner trajectory
            self.current_path = self.smooth_path_bspline(raw_path, num_points=20)
            self.path_waypoint_index = 0
        else:
            self.current_path = None
        
        # Reset goal timer when new goal is selected
        self.goal_start_time = self.step_count
        
        return best_frontier
    
    def navigate_to_frontier(self, goal: Tuple[float, float]) -> CommandsDict:
        """
        Navigate toward a frontier goal using PID control and pure pursuit trajectory tracking.
        Falls back to direct navigation if no path available.
        
        Args:
            goal: (x, y) world coordinates of target
            
        Returns:
            CommandsDict: Movement commands
        """
        command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

        current_pos = self.measured_gps_position()
        current_angle = self.measured_compass_angle()
        if current_pos is None or current_angle is None:
            raise NotImplementedError("Odometry based calculation not implemented yet") # TODO: Odometry based calc
        
        # Pure pursuit: find look-ahead point on smoothed trajectory
        target_world = None
        on_path = False
        
        if self.current_path is not None and len(self.current_path) > 0:
            # Pure pursuit parameters
            LOOK_AHEAD_DIST = 50  # pixels - look ahead distance
            
            # Find closest point on entire remaining path (prevents oscillation from missed waypoints)
            closest_idx = self.path_waypoint_index
            closest_dist = float('inf')
            
            for i in range(self.path_waypoint_index, len(self.current_path)):
                wp = self.current_path[i]
                dx = wp[0] - current_pos[0]
                dy = wp[1] - current_pos[1]
                dist = math.sqrt(dx**2 + dy**2)
                
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = i
            
            # Update waypoint index to closest point (allows skipping ahead if overshot)
            self.path_waypoint_index = closest_idx
            
            # Find look-ahead point starting from closest point
            best_target = None
            best_dist_diff = float('inf')
            
            for i in range(self.path_waypoint_index, len(self.current_path)):
                wp = self.current_path[i]
                dx = wp[0] - current_pos[0]
                dy = wp[1] - current_pos[1]
                dist = math.sqrt(dx**2 + dy**2)
                
                # Find waypoint closest to look-ahead distance
                dist_diff = abs(dist - LOOK_AHEAD_DIST)
                if dist_diff < best_dist_diff:
                    best_dist_diff = dist_diff
                    best_target = wp
                
                # Stop searching too far ahead
                if dist > LOOK_AHEAD_DIST * 1.5:
                    break
            
            if best_target is not None:
                target_world = best_target
                on_path = True
            elif self.path_waypoint_index < len(self.current_path):
                target_world = self.current_path[self.path_waypoint_index]
                on_path = True
            else:
                target_world = goal
        else:
            target_world = goal
        
        # Calculate direction to target
        dx = target_world[0] - current_pos[0]
        dy = target_world[1] - current_pos[1]
        
        # Check for wall obstacles ahead using combined LIDAR+semantic
        wall_obstacle_ahead = self.detect_wall_obstacle(distance_threshold=60.0)

        
        # PID controller for rotation
        target_angle = math.atan2(dy, dx)
        angle_error = normalize_angle(target_angle - current_angle)
        
        # Dead zone for small errors
        DEAD_ZONE = 0.015  # ~0.9 degree - tighter dead zone
        if abs(angle_error) < DEAD_ZONE:
            angle_error = 0.0
            self.angle_error_integral = 0.0  # Reset integral in dead zone
        
        # PID gains - tuned for strict trajectory tracking
        kp = 1.5  # Proportional - increased for faster response
        ki = 0.02  # Integral - slight increase for steady-state accuracy
        kd = 0.5  # Derivative - increased damping to prevent overshoot
        
        # Integral term with anti-windup
        self.angle_error_integral += angle_error
        self.angle_error_integral = max(-8.0, min(8.0, self.angle_error_integral))  # Tighter clamp
        
        # Derivative term
        angle_error_derivative = angle_error - self.last_angle_error
        self.last_angle_error = angle_error
        
        # PID output
        rotation = kp * angle_error + ki * self.angle_error_integral + kd * angle_error_derivative
        rotation = max(-1.0, min(1.0, rotation))
        command["rotation"] = float(rotation)
        
        # Speed control - modulate based on tracking error and wall obstacles
        if wall_obstacle_ahead:
            # Wall detected - slow down significantly
            command["forward"] = 0.2
        elif on_path:
            # Following smooth trajectory - reduce speed with high rotation for tighter tracking
            if abs(rotation) > 0.4:
                command["forward"] = 0.5
            elif abs(rotation) > 0.2:
                command["forward"] = 0.65
            else:
                command["forward"] = 0.75
        else:
            # Direct navigation
            if abs(rotation) > 0.6:
                command["forward"] = 0.4
            else:
                command["forward"] = 0.65
        
        return command
        
    def detect_wall_obstacle(self, distance_threshold: float = 60.0) -> bool:
        """
        Detect if there's a wall obstacle ahead by combining LIDAR and semantic sensor.
        
        Logic: LIDAR detects all objects, semantic detects only drones/wounded/rescue center.
        If LIDAR shows close obstacle but semantic doesn't detect anything at that angle,
        then it must be a wall.
        
        Args:
            distance_threshold: Distance threshold in pixels to consider as obstacle
            
        Returns:
            bool: True if wall obstacle detected within threshold
        """
        lidar_values = self.lidar_values()
        semantic_data = self.semantic_values()
        
        if lidar_values is None:
            return False
        
        # Get lidar sensor for ray angles
        lidar_sensor = self.lidar()
        if lidar_sensor is None:
            return False
        
        ray_angles = lidar_sensor.ray_angles  # type: ignore
        
        # Find rays with obstacles within threshold
        close_obstacles = []
        for i, dist in enumerate(lidar_values):
            if dist < distance_threshold:
                close_obstacles.append((dist, ray_angles[i]))
        
        if not close_obstacles:
            return False  # No close obstacles at all
        
        # Check if any close obstacle is NOT detected by semantic sensor
        # Semantic sensor can only detect: WOUNDED_PERSON, DRONE, RESCUE_CENTER (NOT walls)
        if semantic_data is None or len(semantic_data) == 0:
            # LIDAR sees obstacle but semantic doesn't = must be wall
            return True
        
        # Check each close LIDAR obstacle
        for lidar_dist, lidar_angle in close_obstacles:
            # Check if this obstacle is detected by semantic sensor
            found_in_semantic = False
            for sem_data in semantic_data:
                # Check if semantic detection is at similar angle and distance
                angle_diff = abs(sem_data.angle - lidar_angle)
                # Normalize angle difference to [-pi, pi]
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                
                # If semantic detects something at similar angle/distance, it's not a wall
                if angle_diff < 0.2 and abs(sem_data.distance - lidar_dist) < 30:  # ~11 degrees tolerance
                    found_in_semantic = True
                    break
            
            # If this LIDAR obstacle is NOT in semantic, it's a wall
            if not found_in_semantic:
                return True
        
        return False  # All close obstacles are non-wall entities

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
            
            # Export smoothed B-spline path (already in world coordinates)
            path_world = self.current_path  # Already smoothed and in world coords
            
            data = {
                'occupancy_grid': self.occupancy_grid.copy(),
                'global_occupancy_grid': self.global_occupancy_grid.copy() if self.global_occupancy_grid is not None else None,
                'visited_cells': self.visited_cells.copy(),
                'frontiers': frontiers[:20],  # Top 20 frontiers only
                'current_goal': self.current_frontier_goal,
                'current_path': path_world,  # BFS path in world coordinates
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
        Decentralized main control loop of each drone. 
        Uses a Finite State Machine (FSM) planning logic.

        Returns:
            CommandsDict: The control command for the drone.
        """
        # Increment step counter for timer-based logic
        self.step_count += 1
        
        command: CommandsDict = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # ON FIRST CALL

        # Save initial position and return base
        if self.step_count == 1:
            start_pos = self.measured_gps_position()
            if start_pos is not None:
                self.start_position = (start_pos[0], start_pos[1])
                # Return base is same as start position
                self.return_base = self.start_position
            else:
                pass
                # TODO: Else use odometer if no GPS available to find start position

        # Update occupancy grid from sensors
        self.update_occupancy_grid()
        
        # DEBUG: Export visualization data (only drone 0, periodically)
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
            
            # Only update frontier goal if:
            # 1. No current goal exists, OR
            # 2. Close to current goal (within threshold), OR
            # 3. Timeout exceeded (stuck pursuing same goal too long)
            should_update_goal = False
            GOAL_UPDATE_THRESHOLD = 80.0  # Only update when within 80 pixels of current goal
            
            if self.current_frontier_goal is None:
                should_update_goal = True
            else:
                current_pos = self.measured_gps_position()
                if current_pos is not None:
                    dx = self.current_frontier_goal[0] - current_pos[0]
                    dy = self.current_frontier_goal[1] - current_pos[1]
                    dist_to_goal = math.sqrt(dx**2 + dy**2)
                    if dist_to_goal < GOAL_UPDATE_THRESHOLD:
                        should_update_goal = True
                    
                    # Check for timeout (stuck in loop)
                    elapsed_steps = self.step_count - self.goal_start_time
                    if elapsed_steps > self.goal_timeout_threshold:
                        should_update_goal = True
                        # print(f"Drone {self.identifier}: Goal timeout ({elapsed_steps} steps), selecting new frontier")
            
            # Find new frontier only when needed
            if should_update_goal:
                new_goal = self.select_frontier_goal()
                if new_goal is not None:
                    self.current_frontier_goal = new_goal
            
            # Navigate to frontier 
            if self.current_frontier_goal is not None:
                command = self.navigate_to_frontier(self.current_frontier_goal)
            else:
                # No frontiers available ; random exploration as fallback
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

