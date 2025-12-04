# Designing the Controller

- This is a long-horizon problem with multiple skills and rules to adhere. The combination of all scenarios explore combinatorially making it infeasible to train one single RL policy for the whole event, especially, in multi-agent setting.
- Idea is to come up with a library of policies (programmed or learnt via RL) or primitives and a high-level intelligence (programmed logic or high-level subgoal policy) which chooses a low level policy based on the scenario. This also allows for flexible multi-agent behaviour as dictated by our needs. Meta-learner.
  
## High level policy
- Input: Position, Orientation, Health, grasp state, lidar, semantic, comms message, map, 
- Output: Subgoals: explore, rescue, drop, return base
- If wounded person detected: command rule-based policy to rescue and drop in rescue centre.
- Would need RL policy for task allocation under uncertainty.
  
## Low level policy
- Rule based: 
  - Move to specified point (return base/ rescue centre)
  - Grasp or release
  - Emergency collision avoidance: Rotate by pi (or similar simple logic) for collision avoidance with other drone or wall. Normal possible collision while exploration can be handled by policy inherently.
  - Person pass to other drone logic (maybe an explorer saver mechanism and just pass to saver and move on?)
- RL:
    - Explore unknown areas (since unknown maps). Need RL?


## Scratchpad
- Scenario:
  - Single drone, may be no gps area:
    - At start all drones explore (RL vs programmed. Not sure)
    - If see wounded person, rule based policy to pick and drop in rescue zone. (navigation here might benefit from programmed exploration for collision avoidance paths)
    - If drone in no gps: use odometry for position tracking.
    - Rule based policy to return to base if not enough time left. (how do you judge time it takes for drone to return to decide when to pack?)
    - High level logic:
      - Keep exploring always (learnt collision avoidance)
      - If person found issue rescue and drop policy
      - If times almost out issue return to base policy
      - Wall collision avoidance
      - programmed collision avoidance for use during programmed policies.
  - TODO:
    - Exploration policy (programmed)
    - RL exploration policy
    - If found person - move to person for pick up- programmed policy.
    - Pick person and drop at rescue centre programmed policy
    - Return to base when time's almost out programmed policy.
    - High level logic policy
   
    - multiagent
    - Write logic to combine maps of multiple drones
    - communicate the locations where persons are
    - drones should pathplan and go to that location to rescue persons.

