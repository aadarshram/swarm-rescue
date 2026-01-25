"""
Real-time Occupancy Grid and Frontier Visualization
Displays the occupancy grid, frontiers, and drone positions in a separate matplotlib window
Run this in parallel with the simulation to see live updates
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation
import pickle
import os
from pathlib import Path

class OccupancyGridVisualizer:
    def __init__(self, update_interval=500, show_global=True):
        """
        Initialize the visualizer
        
        Args:
            update_interval: Update interval in milliseconds
            show_global: If True, show global grid (right); if False, show visit heatmap
        """
        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 8))
        self.ax_grid = self.axes[0]
        self.ax_right = self.axes[1]
        
        self.update_interval = update_interval
        self.show_global = show_global
        self.data_file = Path("/tmp/drone_grid_data.pkl")
        
        # Initialize plot elements
        self.grid_img = None
        self.global_img = None
        self.visits_img = None
        self.colorbar = None
        self.frontier_scatter = None
        self.goal_scatter = None
        self.drone_scatter = None
        self.path_line = None  # BFS path visualization
        
        # Track map dimensions to detect changes
        self.last_extent = None
        self.last_timestamp = None
        
        # Setup plots
        self.setup_plots()
        
    def setup_plots(self):
        """Setup the plot axes and labels"""
        self.ax_grid.set_title("Local Occupancy Grid (Drone 0)\n(Gray=Unknown, White=Free, Black=Occupied, Red=Wounded, Green=Rescue)", 
                              fontsize=10)
        self.ax_grid.set_xlabel("X (pixels)")
        self.ax_grid.set_ylabel("Y (pixels)")
        self.ax_grid.grid(True, alpha=0.3)
        
        if self.show_global:
            self.ax_right.set_title("Global Occupancy Grid (All Drones)\n(Gray=Unknown, White=Free, Black=Occupied, Red=Wounded, Green=Rescue)", 
                                   fontsize=10)
        else:
            self.ax_right.set_title("Visit Frequency Heatmap", fontsize=10)
        self.ax_right.set_xlabel("X (pixels)")
        self.ax_right.set_ylabel("Y (pixels)")
        self.ax_right.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def load_data(self):
        """Load data from shared file"""
        try:
            if self.data_file.exists():
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                return data
        except Exception as e:
            # File might be being written, try next update
            pass
        return None
    
    def update(self, frame):
        """Update function called by animation"""
        data = self.load_data()
        
        if data is None:
            return []
        
        # Extract data
        occupancy_grid = data.get('occupancy_grid')
        global_occupancy_grid = data.get('global_occupancy_grid')
        visited_cells = data.get('visited_cells')
        frontiers = data.get('frontiers', [])
        current_goal = data.get('current_goal')
        current_path = data.get('current_path')  # BFS path
        drone_positions = data.get('drone_positions', [])
        grid_origin = data.get('grid_origin', (0, 0))
        grid_resolution = data.get('grid_resolution', 8)
        timestamp = data.get('timestamp', 0)
        
        if occupancy_grid is None:
            return []
        
        # Calculate extent for this map
        grid_height, grid_width = occupancy_grid.shape
        current_extent = [grid_origin[0], 
                         grid_origin[0] + grid_width * grid_resolution,
                         grid_origin[1], 
                         grid_origin[1] + grid_height * grid_resolution]
        
        # Detect new run: timestamp reset to 0 or extent changed
        new_run_detected = False
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            new_run_detected = True
        if self.last_extent is not None and self.last_extent != current_extent:
            new_run_detected = True
        
        # Reset visualization for new run
        if new_run_detected:
            self.reset_plots()
        
        self.last_timestamp = timestamp
        self.last_extent = current_extent
        
        # Create RGB image for occupancy grid
        rgb_grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Color mapping: UNKNOWN=0, FREE=1, OCCUPIED=2, WOUNDED=3, RESCUE=4
        rgb_grid[occupancy_grid == 0] = [128, 128, 128]  # Gray - Unknown
        rgb_grid[occupancy_grid == 1] = [255, 255, 255]  # White - Free
        rgb_grid[occupancy_grid == 2] = [0, 0, 0]        # Black - Occupied
        rgb_grid[occupancy_grid == 3] = [255, 0, 0]      # Red - Wounded
        rgb_grid[occupancy_grid == 4] = [0, 255, 0]      # Green - Rescue
        
        # Display occupancy grid
        if self.grid_img is None:
            self.grid_img = self.ax_grid.imshow(rgb_grid, origin='lower', extent=current_extent)
            self.ax_grid.set_xlim(current_extent[0], current_extent[1])
            self.ax_grid.set_ylim(current_extent[2], current_extent[3])
        else:
            self.grid_img.set_data(rgb_grid)
            self.grid_img.set_extent(current_extent)
            self.ax_grid.set_xlim(current_extent[0], current_extent[1])
            self.ax_grid.set_ylim(current_extent[2], current_extent[3])
        
        # Display global grid or visit heatmap on right panel
        if self.show_global and global_occupancy_grid is not None:
            # Create RGB image for global occupancy grid
            global_rgb_grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            
            # Color mapping: UNKNOWN=0, FREE=1, OCCUPIED=2, WOUNDED=3, RESCUE=4
            global_rgb_grid[global_occupancy_grid == 0] = [128, 128, 128]  # Gray - Unknown
            global_rgb_grid[global_occupancy_grid == 1] = [255, 255, 255]  # White - Free
            global_rgb_grid[global_occupancy_grid == 2] = [0, 0, 0]        # Black - Occupied
            global_rgb_grid[global_occupancy_grid == 3] = [255, 0, 0]      # Red - Wounded
            global_rgb_grid[global_occupancy_grid == 4] = [0, 255, 0]      # Green - Rescue
            
            if self.global_img is None:
                self.global_img = self.ax_right.imshow(global_rgb_grid, origin='lower', extent=current_extent)
                self.ax_right.set_xlim(current_extent[0], current_extent[1])
                self.ax_right.set_ylim(current_extent[2], current_extent[3])
            else:
                self.global_img.set_data(global_rgb_grid)
                self.global_img.set_extent(current_extent)
                self.ax_right.set_xlim(current_extent[0], current_extent[1])
                self.ax_right.set_ylim(current_extent[2], current_extent[3])
                
            # Update title with global grid stats
            total_cells = grid_width * grid_height
            g_unknown = np.sum(global_occupancy_grid == 0)
            g_free = np.sum(global_occupancy_grid == 1)
            g_occupied = np.sum(global_occupancy_grid == 2)
            g_explored_pct = ((g_free + g_occupied) / total_cells) * 100
            
            self.ax_right.set_title(
                f"Global Grid (All Drones) - Explored: {g_explored_pct:.1f}%\n"
                f"Unknown: {g_unknown}, Free: {g_free}, Occupied: {g_occupied}", 
                fontsize=10
            )
            
        # Display visit heatmap
        elif not self.show_global and visited_cells is not None:
            visits_display = np.log1p(visited_cells)  # Log scale for better visualization
            if self.visits_img is None:
                self.visits_img = self.ax_right.imshow(visits_display, origin='lower', 
                                                        cmap='hot', extent=current_extent)
                self.colorbar = plt.colorbar(self.visits_img, ax=self.ax_right, label='log(visits+1)')
                self.ax_right.set_xlim(current_extent[0], current_extent[1])
                self.ax_right.set_ylim(current_extent[2], current_extent[3])
            else:
                self.visits_img.set_data(visits_display)
                self.visits_img.set_extent(current_extent)
                self.visits_img.set_clim(vmin=0, vmax=visits_display.max())
                self.ax_right.set_xlim(current_extent[0], current_extent[1])
                self.ax_right.set_ylim(current_extent[2], current_extent[3])
        
        # Clear previous frontier/goal/drone/path markers
        if self.frontier_scatter is not None:
            self.frontier_scatter.remove()
            self.frontier_scatter = None
        if self.goal_scatter is not None:
            self.goal_scatter.remove()
            self.goal_scatter = None
        if self.drone_scatter is not None:
            self.drone_scatter.remove()
            self.drone_scatter = None
        if self.path_line is not None:
            self.path_line.remove()
            self.path_line = None
        
        # Plot frontiers on both axes
        if frontiers:
            frontier_x = [f[0][0] for f in frontiers]
            frontier_y = [f[0][1] for f in frontiers]
            self.frontier_scatter = self.ax_grid.scatter(frontier_x, frontier_y, 
                                                         c='cyan', s=50, marker='x', 
                                                         label='Frontiers', alpha=0.7,
                                                         linewidths=2)
        
        # Plot BFS path if available
        if current_path is not None and len(current_path) > 0:
            path_x = [p[0] for p in current_path]
            path_y = [p[1] for p in current_path]
            self.path_line, = self.ax_grid.plot(path_x, path_y, 
                                                 'g-', linewidth=2, alpha=0.6,
                                                 label='BFS Path')
            # Also plot path waypoints as small dots
            self.ax_grid.scatter(path_x, path_y, 
                                c='lime', s=20, marker='o', 
                                alpha=0.5, zorder=5)
        
        # Plot current goal
        if current_goal is not None:
            self.goal_scatter = self.ax_grid.scatter([current_goal[0]], [current_goal[1]], 
                                                     c='yellow', s=200, marker='*', 
                                                     label='Current Goal', 
                                                     edgecolors='black', linewidths=2)
        
        # Plot drone positions
        if drone_positions:
            drone_x = [p[0] for p in drone_positions]
            drone_y = [p[1] for p in drone_positions]
            self.drone_scatter = self.ax_grid.scatter(drone_x, drone_y, 
                                                      c='blue', s=100, marker='o', 
                                                      label='Drones', 
                                                      edgecolors='white', linewidths=2)
        
        # Update legend
        if self.frontier_scatter or self.goal_scatter or self.drone_scatter or self.path_line:
            self.ax_grid.legend(loc='upper right', fontsize=8)
        
        # Update title with stats
        if occupancy_grid is not None:
            total_cells = grid_width * grid_height
            unknown = np.sum(occupancy_grid == 0)
            free = np.sum(occupancy_grid == 1)
            occupied = np.sum(occupancy_grid == 2)
            explored_pct = ((free + occupied) / total_cells) * 100
            
            self.ax_grid.set_title(
                f"Occupancy Grid - Explored: {explored_pct:.1f}%\n"
                f"Unknown: {unknown}, Free: {free}, Occupied: {occupied}\n"
                f"Frontiers: {len(frontiers)}, Drones: {len(drone_positions)}", 
                fontsize=10
            )
        
        return []
    
    def reset_plots(self):
        """Reset all plot elements for a new run"""
        # Clear images
        if self.grid_img is not None:
            self.grid_img.remove()
            self.grid_img = None
        if self.global_img is not None:
            self.global_img.remove()
            self.global_img = None
        if self.visits_img is not None:
            self.visits_img.remove()
            self.visits_img = None
        
        # Remove colorbar properly
        if self.colorbar is not None:
            try:
                self.colorbar.ax.remove()  # Remove the colorbar axes
            except:
                pass
            self.colorbar = None
        
        # Clear markers
        if self.frontier_scatter is not None:
            self.frontier_scatter.remove()
            self.frontier_scatter = None
        if self.goal_scatter is not None:
            self.goal_scatter.remove()
            self.goal_scatter = None
        if self.drone_scatter is not None:
            self.drone_scatter.remove()
            self.drone_scatter = None
        if self.path_line is not None:
            self.path_line.remove()
            self.path_line = None
        
        # Clear axes and reset
        self.ax_grid.clear()
        self.ax_right.clear()
        self.setup_plots()
        
        print("="*60)
        print("NEW RUN DETECTED - Visualizer Reset")
        print("="*60)
    
    def run(self):
        """Start the visualization"""
        print("="*60)
        print("Occupancy Grid & Frontier Visualizer")
        print("="*60)
        print(f"Waiting for simulation data at: {self.data_file}")
        print("Run the simulation with my_drone_FSM to see live updates")
        if self.show_global:
            print("Mode: Showing GLOBAL occupancy grid (right) - shared by all drones")
        else:
            print("Mode: Showing visit frequency heatmap (right)")
        print("Close this window to exit")
        print("="*60)
        
        anim = FuncAnimation(self.fig, self.update, interval=self.update_interval, 
                           blit=False, cache_frame_data=False)
        plt.show()


if __name__ == "__main__":
    # Set show_global=True to show global occupancy grid, False for visit heatmap
    visualizer = OccupancyGridVisualizer(update_interval=500, show_global=True)
    visualizer.run()
