# AI Assistant Instructions for Swarm-Rescue

This document guides AI coding assistants in working with the Swarm-Rescue codebase.

## Project Overview

Swarm-Rescue is a simulation environment for programming autonomous drone swarms to rescue injured people in disaster scenarios. Key characteristics:

- Built on Simple-Playgrounds (SPG) using Pymunk physics and Arcade game engine
- Implements realistic physics (mass, inertia, collisions)
- Focuses on swarm coordination and autonomous decision-making
- Scoring based on rescued people, exploration coverage, drone health, and time efficiency

## Core Architecture

### Key Components

1. **Drone Controller** (`src/swarm_rescue/solutions/my_drone_eval.py`)
   - Main entry point for custom drone behavior
   - Inherits from base classes that provide sensor access and movement capabilities

2. **Simulation Engine** (`src/swarm_rescue/simulation/`)
   - `drone/`: Drone physics and controls
   - `elements/`: Environment objects (walls, people, etc.)
   - `ray_sensors/`: LIDAR and semantic sensor implementations
   - `reporting/`: Scoring and evaluation tools

3. **Maps** (`src/swarm_rescue/maps/`)
   - Contains predefined and competition maps
   - Each map defines wall layouts and object placements

## Development Workflow

1. **Setup Environment**
   ```bash
   # Install dependencies (Ubuntu recommended)
   pip install -e .
   ```

2. **Implement Drone Logic**
   - Extend `MyDroneEval` in `solutions/my_drone_eval.py`
   - Use provided sensors:
     - LIDAR: 360° field of view, 181 rays, 300px range
     - Semantic sensor: Object type detection
     - GPS and compass for absolute positioning
     - Odometer for relative positioning

3. **Testing**
   - Use example maps for initial testing
   - Progress through intermediate to competition maps
   - Monitor drone health and collision handling

## Key Patterns & Conventions

1. **Sensor Usage**
   - LIDAR data is noisy by design (Gaussian noise added)
   - First (-Pi) and last (Pi) LIDAR readings should match (360° view)
   - See `examples/example_display_lidar.py` for visualization

2. **Performance Considerations**
   - GPU-accelerated sensor computations via OpenGL shaders
   - Time limits enforce both simulation steps and wall-clock time
   - Optimize for both exploration speed and computational efficiency

3. **Map Exploration**
   - Implement efficient area coverage strategies
   - Balance exploration vs rescue priorities
   - Handle communication losses and drone failures gracefully

## Common Pitfalls

1. **Environment Compatibility**
   - Avoid VirtualBox - OpenGL shader issues
   - macOS not officially supported (Metal vs OpenGL)
   - See troubleshooting in `INSTALL.md` for OpenGL issues

2. **Code Organization**
   - Keep drone logic modular and maintainable
   - Avoid tight coupling to specific map layouts
   - Test with multiple map configurations

## Testing Notes

- All development must be in simulation environment
- Final evaluation uses unknown maps
- Scoring considers multiple factors beyond just rescue count