# Real-Time Occupancy Grid Visualization

View your drone's exploration in real-time with live occupancy grid and frontier visualization!

## Quick Start

### Option 1: Simple Launch (Recommended)

```bash
# Terminal 1 - Start visualizer first
python src/swarm_rescue/solutions/launch_visualizer.py

# Terminal 2 - Run simulation
python src/swarm_rescue/launcher.py
```

### Option 2: Direct Launch

```bash
# Terminal 1
python src/swarm_rescue/solutions/occupancy_grid_visualizer.py

# Terminal 2
python src/swarm_rescue/launcher.py
```

## What You'll See

### Left Panel: Occupancy Grid
- **Gray**: Unknown (unexplored)
- **White**: Free space (confirmed empty)
- **Black**: Occupied (walls/obstacles)
- **Red**: Wounded person location
- **Green**: Rescue center location
- **Cyan X markers**: Detected frontiers
- **Yellow star**: Current goal
- **Blue circles**: Drone positions

### Right Panel: Visit Frequency Heatmap
- Shows how many times each area has been visited
- Hot colors (red/yellow) = frequently visited
- Cool colors (dark) = rarely visited
- Helps identify if drone is exploring efficiently

## Features

✅ **Live Updates**: Refreshes every 500ms  
✅ **Exploration Stats**: Shows % explored, frontier count  
✅ **Multi-drone Support**: Shows all drone positions  
✅ **Visit Tracking**: Heatmap shows exploration patterns  
✅ **Frontier Visualization**: See where drone plans to go next  
✅ **No Simulation Impact**: Runs in separate process  

## Data Location

The visualization reads from: `/tmp/drone_grid_data.pkl`

This file is automatically created and updated by Drone 0 during simulation.

## Troubleshooting

**No data showing?**
- Make sure simulation is running with `my_drone_FSM`
- Wait a few seconds for first data export
- Check that Drone 0 is initialized

**Window freezing?**
- Close and restart the visualizer
- Increase update interval in `occupancy_grid_visualizer.py` (line 179)

**Performance issues?**
- Reduce `viz_export_interval` in `my_drone_FSM.py` (currently 5)
- Close visualizer if not needed

## Customization

Edit `my_drone_FSM.py`:
```python
self.viz_export_interval = 10  # Export less frequently (every 10 cycles)
```

Edit `occupancy_grid_visualizer.py`:
```python
update_interval=1000  # Update visualization every 1000ms instead of 500ms
```

## Technical Details

- **Export**: Only Drone 0 exports data to avoid file conflicts
- **Format**: Pickle serialization for fast I/O
- **Atomic Writes**: Uses temp file + rename for safe concurrent access
- **Data Size**: ~50-100KB per export (depends on map size)
- **Update Rate**: 5 control cycles (~100ms simulation time)

## Files

- `occupancy_grid_visualizer.py` - Main visualization code
- `launch_visualizer.py` - Convenience launcher
- `my_drone_FSM.py` - Modified to export data (lines with `export_visualization_data`)

Enjoy exploring your drone's cognitive map in real-time! 🚁
