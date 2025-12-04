# Available Reward Signals for RL Training

This document lists all environment variables and drone properties that can be used for reward calculation in the `_calculate_reward()` method of `DroneRLEnv`.

## 1. Drone State Variables

### Position & Orientation (# Use noisy values)
- **`self._agent.true_position()`** → `np.ndarray[x, y]`
  - True position in pixels (ground truth)
  - Use for: Distance calculations, position-based rewards
  
- **`self._agent.true_angle()`** → `float` (radians, -π to π)
  - True orientation angle
  - Use for: Heading-based rewards, angular penalties
  
- **`self._agent.true_velocity()`** → `np.ndarray[vx, vy]`
  - True velocity in pixels/second
  - Use for: Speed penalties/rewards, momentum tracking
  
- **`self._agent.true_angular_velocity()`** → `float` (radians/second)
  - True angular velocity
  - Use for: Rotation penalties (excessive spinning)

### Health & Collisions
- **`self._agent.drone_health`** → `int` (0 to `DRONE_INITIAL_HEALTH`)
  - Current health (reduced by collisions)
  - Use for: Collision penalties, survival rewards
  - Initial value: `constants.DRONE_INITIAL_HEALTH`
  
- **`self._agent.removed`** → `bool`
  - Whether drone was destroyed/removed
  - Use for: Large terminal penalty for destruction

### Grasping & Rescue
- **`self._agent.grasped_wounded_persons()`** → `List[WoundedPerson]`
  - List of wounded persons currently grasped
  - Use for: Grasping success rewards
  
- **`self._agent.touch_human()`** → `bool`
  - Whether drone is touching a wounded person (from semantic sensor)
  - Use for: Proximity rewards, grasping opportunities
  
- **`self._agent.reward`** → `float`
  - Accumulated reward from rescued persons (set by RescueCenter)
  - Each rescue adds `1.0` to this value
  - Use for: Tracking successful rescues

### Sensors
- **`self._agent.lidar_values()`** → `np.ndarray` or `None`
  - Array of distances (180 rays in default config)
  - Use for: Collision avoidance rewards, proximity penalties
  
- **`self._agent.semantic_values()`** → Semantic data or `None`
  - Detects: wounded persons, drones, rescue center
  - Format: List of (entity_type, distance, angle, is_grasped)
  - Use for: Navigation rewards, target proximity

### Sensor Status (disabled zones)
- **`self._agent.gps_is_disabled()`** → `bool`
- **`self._agent.compass_is_disabled()`** → `bool`
- **`self._agent.lidar_is_disabled()`** → `bool`
- **`self._agent.semantic_is_disabled()`** → `bool`
- **`self._agent.communicator_is_disabled()`** → `bool`
  - Use for: Penalties when entering disabled zones

### Time Tracking
- **`self._agent.elapsed_timestep`** → `int`
  - Number of simulation steps elapsed
  - Use for: Time efficiency rewards/penalties
  
- **`self._agent.elapsed_walltime`** → `float` (seconds)
  - Real-world time elapsed
  - Use for: Time-based termination

### Return Area
- **`self._agent.is_inside_return_area`** → `bool`
  - Whether drone is in designated return zone
  - Use for: Return bonus, mission completion

## 2. Map & Environment Variables

### Wounded Persons
- **`self._map._number_wounded_persons`** → `int`
  - Total number of wounded persons in map
  - Use for: Calculating rescue progress percentage
  
- **`self._map._wounded_persons_pos`** → `List[tuple]` (if available)
  - Positions of all wounded persons
  - Use for: Distance-based navigation rewards (careful: this is ground truth!)

### Rescue Center
- **`self._map._rescue_center_pos`** → `tuple` (if available)
  - Position of rescue center
  - Use for: Distance rewards for carrying wounded persons

### Exploration
- **`self._map.explored_map.score()`** → `float` (0.0 to 1.0)
  - Percentage of map explored
  - Use for: Exploration rewards (already used in your code)
  - Updates as drones move through map
  
- **`self._map.explored_map.zones_explored`** → exploration data
  - Detailed exploration statistics
  - Use for: Fine-grained exploration rewards

### Episode Stats (custom tracking)
- **`self.episode_stats`** → `dict`
  - Your custom dictionary for tracking episode metrics:
    - `'steps_taken'`: Current step count
    - `'total_reward'`: Cumulative reward
    - `'rescued'`: Number rescued this episode
    - `'success'`: Mission completion flag
    - `'health_lost'`: Total health damage
    - `'exploration_progress'`: Cumulative exploration gain

### Map Properties
- **`self.map_size`** → `tuple` (width, height)
  - Size of the map in pixels
  - Use for: Normalizing positions, boundary checks

## 3. Action Variables

### Current Action
- **`action`** parameter in `_calculate_reward()`
  - 3D array: `[forward, lateral, rotation]`
  - Each in range [-1, 1]
  - Use for: Action penalties (e.g., excessive rotation, high speeds)

## 4. Reward Design Patterns

### Basic Survival
```python
# Time penalty (encourage efficiency)
reward -= 1.0  # Per timestep

# Health penalty
health_loss = (prev_health - current_health) / constants.DRONE_INITIAL_HEALTH
reward -= 10.0 * health_loss

# Destruction penalty (terminal)
if self._agent.removed:
    reward -= 100.0
```

### Collision Avoidance
```python
# LIDAR-based proximity penalty
min_distance = np.min(self._agent.lidar_values())
if min_distance < threshold:
    reward -= (threshold - min_distance) / threshold
```

### Exploration
```python
# Already implemented
exp_progress = current_score - self.last_exp_score
reward += 5.0 * exp_progress
```

### Rescue Progress
```python
# Touch wounded person
if self._agent.touch_human():
    reward += 10.0

# Successfully grasped
if len(self._agent.grasped_wounded_persons()) > 0:
    reward += 5.0  # Per step while carrying

# Successful rescue (check agent.reward increase)
if self._agent.reward > prev_reward:
    reward += 50.0  # Big bonus for delivery
```

### Navigation
```python
# Distance to nearest wounded person (requires position access)
if hasattr(self._map, '_wounded_persons_pos'):
    wounded_positions = self._map._wounded_persons_pos
    drone_pos = self._agent.true_position()
    distances = [np.linalg.norm(drone_pos - wp) for wp in wounded_positions]
    min_dist = min(distances)
    # Reward getting closer
    if min_dist < prev_min_dist:
        reward += 0.1 * (prev_min_dist - min_dist)
```

### Action Regularization
```python
# Rotation penalty (reduce spinning)
reward -= 0.01 * abs(action[2])

# Speed penalty (smooth movement)
reward -= 0.001 * (action[0]**2 + action[1]**2)
```

### Mission Completion
```python
# All wounded rescued
if self.total_rescued >= self._map._number_wounded_persons:
    reward += 100.0
    
# Return to base with all rescued
if self._agent.is_inside_return_area and mission_complete:
    reward += 50.0
```

## 5. Important Notes

### Ground Truth vs Sensors
- **DO NOT** use true position/angle in your control logic
- **CAN** use true values for reward calculation (RL is allowed to see ground truth)
- Sensors (GPS, compass, lidar, semantic) are what the policy sees in observations

### Reward Scaling
- Keep rewards on similar scales (use normalization)
- Large terminal rewards (rescue, destruction) should dominate
- Small step penalties keep agent moving
- Balance exploration vs exploitation

### Common Pitfalls
1. **Sparse rewards**: Add dense shaping rewards (distance, exploration)
2. **Reward hacking**: Agent finds unintended shortcuts
3. **Scale mismatch**: One reward dominates all others
4. **Reward drift**: Cumulative rewards grow unbounded

## 6. Current Reward Implementation

Your current `_calculate_reward()` uses:
- ✅ Timestep penalty: `-1`
- ✅ Rotation penalty: `-0.01 * abs(action[2])`
- ✅ Health loss: Normalized by initial health
- ✅ LIDAR collision avoidance: Proximity penalty < 30px
- ✅ Exploration progress: `5.0 * exp_progress`
- ✅ Touch human: `+20.0`
- ❌ Destruction penalty: Incomplete (syntax error in code)

### Suggested Additions:
1. **Rescue completion bonus**: Large reward when `agent.reward` increases
2. **Distance shaping**: Reward approaching wounded persons
3. **Return bonus**: Reward returning to base after rescues
4. **Survival bonus**: Small positive reward per step alive
5. **Mission completion**: Large terminal reward for rescuing all wounded
