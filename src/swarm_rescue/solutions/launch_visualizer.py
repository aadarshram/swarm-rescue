#!/usr/bin/env python3
"""
Launch script for real-time occupancy grid visualization
Run this BEFORE or DURING the simulation to see live updates
"""

import sys
import subprocess
from pathlib import Path

def main():
    viz_script = Path(__file__).parent / "occupancy_grid_visualizer.py"
    
    if not viz_script.exists():
        print(f"Error: Visualizer script not found at {viz_script}")
        sys.exit(1)
    
    print("="*60)
    print("Starting Occupancy Grid Visualizer")
    print("="*60)
    print("\nInstructions:")
    print("1. Keep this window open")
    print("2. In another terminal, run: python src/swarm_rescue/launcher.py")
    print("3. Watch the real-time grid updates!")
    print("\nThe visualization shows:")
    print("  - Left panel: Occupancy grid with frontiers and current goal")
    print("  - Right panel: Visit frequency heatmap")
    print("\nPress Ctrl+C or close the window to exit")
    print("="*60 + "\n")
    
    try:
        subprocess.run([sys.executable, str(viz_script)])
    except KeyboardInterrupt:
        print("\n\nVisualization stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
