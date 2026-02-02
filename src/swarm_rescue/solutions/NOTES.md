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


-------------------------------------------------
meet on 23.1.26

- proper frontier exploration algorithm
  - resolution. obstacles and new frontiers
  - End of lidar also marked as obstacle. Fix.
  - Frequency map is stupid. Fix.
- Once person picked,
  - Goal directed obstacle avoidance path.
  - Anup BUG 1 BUG 2
- Multi agent fighting. Fix.
  - Picking same person both.
- Going back to rescue center, proper find path. Dont get stuck.

Work:

- Anup: Goal oriented obstacle avoidance AND Going back to rescue center, proper find path. Dont get stuck.
- Arnav and Advay: Multi agent cooperation AND all things multiagent - do research.
- Aadarsh and Kairav: Frontier fix- Kairav focus on good code from online. Aadarsh on existing code to fix. Noise filter.


Mission swarm
- Get 1st in india.
- Finals is on March 19th.
- Jan 23 - catchup
- Jan 30 - single agent full complete- easy medium
- Feb 6 - multi agent easy , single agent hard
- Feb 13 - multi agent medium , single agent hard - all specials
- Feb 20 - multi agent hard  - specials
- Feb 27 - refine 
- March 6 - refine
- March 13 - refine



-----------------------------------------
- traj is not very smooth and I am doing PID following that traj
- After final waypoint still didnt pass frontier goal, stuck looping at final state
- After picking wounded person, implement proper nav to rescue center



Checkpoint 1 results: 1st two environments
count_reachable v3 = 363463
        * Round n°1/2: 
                rescued nb: 1/1, explor. score: 95.8%, health return score: 94.0%, walltime elapsed: 45s/270s, elapse timestep: 1337/2700 steps, time to rescue all: 857 steps.
                percentage of drones destroyed: 0.0 %, mean percentage of drones health : 0.0 %.
                round score: 88.0%, frequency: 29.80 steps/s.

count_reachable v3 = 363463
        * Round n°2/2: 
                rescued nb: 1/1, explor. score: 99.8%, health return score: 92.0%, walltime elapsed: 42s/270s, elapse timestep: 1229/2700 steps, time to rescue all: 860 steps.
                percentage of drones destroyed: 0.0 %, mean percentage of drones health : 0.0 %.
                round score: 95.2%, frequency: 29.50 steps/s.



----------------
TODO for Jan 31st:
- DONE: In the interpolation logic for anything adaptive, instead of uniform scaling do exponential tail of sorts. For example, in velocity control to prevent collision, as distance gets smaller exponentially decrease the speed at a rate. But linearly increase. This is so that even though inertia, it will try to vigorously stop.
- Implement closed loop control for gettign back to rescue center. Since u dont know how many waypoints to rescue center u need to track segments with enough like how frontier is.
- Rescue center posiiton is sometimes not found out and or sometimes is not in the right place. Find why.
- Implement return to base using similar logic as that of rescue center.




----------------------------------------------------------------
Meet on Feb 2: 2/2/26

Notes:
- While moving with human, it may be slow.
- Rescue center position identified wrong
- bfs path back to rescue center sometimes not found.
- no reachable frontier and goes to random explore in between fidning frontier. idk why. but it should find.
- optimizing speed- occupancy grid
- write code to make lidar find if small obstacle- if so go to semantic sensor range.
- some way to identify narrow spaces and put special logic.
- for frontiers in multi drone- do nearest drone allocation
- Try plotting bfs path without smoothening and with smoothening. Si smoothining needed?
- race conditions eliminate- shoudl back off- do across time, no race again.
- frontier goal distance buffer- narrow space hoga?
- frontier distance check no L2 do, try line of sight distance
- Ramer–Douglas–Peucker (RDP)- it reduces waypoints to find a better path- but does not cosnider obstacle. we need to consider obstacle.
- go through BFS deeper- why path not correct? lil short do. Arnav wants waypoint lil do near wall corners.
- but ad wants smooth for speed. speed VS distance tradeoff
- make algo based on search VS drop person.
- drone colliison use inflated wall but then problem in narrow spaces.
- premature return to base, cuz frontiers exist but not reachable.
- if frontier not reachable, reach as best as possible.
- closed loop trip back to rescue center. and back to base
- real time other drone avoidance - lil bit angle turn same direction.
- speed efficiency - is it occupancy grid?







