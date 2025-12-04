import numpy as np
import random  # important for cone pattern randomness
from highway_env.envs.highway_env import HighwayEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane
from highway_env.vehicle.behavior import IDMVehicle


class HighwayConstructionEnv(HighwayEnv):
    """
    FINAL optimized construction environment.

    Features:
    - Ego starts in random lane among [0,1,2]
    - 18 traffic vehicles at near-constant speed (BASE_SPEED ± 0.5 m/s)
    - Ego speed = traffic_speed + 1 m/s
    - Random cone pattern each episode (but fixed lane=0)
    - NO slow cars
    - DRL-friendly shaped reward with:
        * speed shaping
        * lane preference
        * construction zone shaping
        * safe-distance shaping
        * center-of-lane stabilization
        * stronger forward-progress reward
        * clipped rewards for stability
    """

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update({
            "lanes_count": 4,
            "vehicles_count": 18,       # medium difficulty
            "duration": 120,
            "screen_width": 1200,
            "screen_height": 400,
            "scaling": 5.5,
            "collision_reward": -50,    # softened crash penalty
            "highway_length": 1000,
            "centering_position": [0.3, 0.5],
            "simulation_frequency": 10,
            "policy_frequency": 5,
        })
        return cfg

    # --------------------------------------------------
    # ROAD SETUP
    # --------------------------------------------------
    def _create_road(self):
        net = RoadNetwork()
        lane_w = 4.0
        L = self.config["highway_length"]

        # Straight 4-lane highway
        for i in range(self.config["lanes_count"]):
            origin = np.array([0.0, i * lane_w])
            end = np.array([L, i * lane_w])
            net.add_lane("a", "b", StraightLane(origin, end, width=lane_w))

        # Entrance ramp
        net.add_lane("entrance", "a",
                     StraightLane(np.array([L * 0.05, -lane_w * 2]),
                                  np.array([L * 0.2, lane_w]), width=lane_w))

        # Exit ramp
        net.add_lane("b", "exit",
                     StraightLane(np.array([L * 0.8, 0]),
                                  np.array([L * 0.9, -lane_w * 2]), width=lane_w))

        self.road = Road(net, np_random=self.np_random)

    # --------------------------------------------------
    # VEHICLE SPAWNING
    # --------------------------------------------------
    def _create_vehicles(self):
        from highway_env.vehicle.controller import ControlledVehicle

        BASE_SPEED = 30.0  # approx. 67 mph

        # Ego in random lane among [0,1,2]
        start_lane = self.np_random.choice([0, 1, 2])
        ego_lane = self.road.network.get_lane(("a", "b", start_lane))

        # Ego always slightly faster than traffic
        ego_speed = BASE_SPEED + 1.0

        ego_vehicle = ControlledVehicle(self.road,
                                        ego_lane.position(50, 0),
                                        speed=ego_speed)
        self.vehicle = ego_vehicle
        self.road.vehicles.append(ego_vehicle)

        # Traffic: constant speed ±0.5 m/s for natural variation
        for _ in range(self.config["vehicles_count"]):
            lane = self.np_random.choice(self.road.network.lanes_list())
            pos = self.np_random.uniform(0, 300)  # traffic near ego for interaction
            x, y = lane.position(pos, 0)

            speed = BASE_SPEED + self.np_random.uniform(-0.5, 0.5)
            self.road.vehicles.append(IDMVehicle(self.road, [x, y], speed=speed))

        # --------------------------------------------------
        # RANDOMIZED CONSTRUCTION PATTERN in lane 0
        # --------------------------------------------------
        cone_lane_idx = 0
        cone_lane = self.road.network.get_lane(("a", "b", cone_lane_idx))

        # Patterns of different lengths—use Python random.choice
        cone_patterns = [
            [390, 400, 410, 420, 430, 440],
            [395, 410, 425, 440, 455],
            [400, 415, 430, 445],
            [385, 405, 425, 445]
        ]
        cone_offsets = random.choice(cone_patterns)

        # Place cones
        for offset in cone_offsets:
            x, y = cone_lane.position(offset, 0)
            cone = IDMVehicle(self.road, [x, y], speed=0, target_speed=0)
            cone.color = (255, 120, 0)
            self.road.vehicles.append(cone)

    # --------------------------------------------------
    # REWARD FUNCTION (FINAL OPTIMIZED)
    # --------------------------------------------------
    def _reward(self, action):
        v = self.vehicle
        r = 0.0

        mph = v.speed / 0.44704
        pos_x = v.position[0]
        lane_idx = v.lane_index[2] if hasattr(v, "lane_index") else 1

        # 1. Crash penalty
        if v.crashed:
            return -50.0

        # 2. Survival reward
        r += 0.1

        # 3. Speed shaping (optimal around 65 mph)
        lower, upper, optimal = 60, 70, 65
        if lower <= mph <= upper:
            r += 1.0
        else:
            r -= 0.02 * abs(mph - optimal)

        # 4. Preferred lanes (1 & 2), penalty for lane 3
        if not (380 < pos_x < 480):
            if lane_idx in [1, 2]:
                r += 0.2
            elif lane_idx == 3:
                r -= 0.05

        # 5. Smart lane-changing
        if action in [3, 4]:  # left/right
            front, _ = v.road.neighbour_vehicles(v, v.lane_index)
            if front:
                r += 0.1
            else:
                r -= 0.05

        # 6. Construction zone shaping (softer penalty)
        if 380 < pos_x < 480:
            if lane_idx == 0:
                r -= 2.5
            else:
                r += 1.0

        # 7. Safe-distance shaping
        front, _ = v.road.neighbour_vehicles(v, v.lane_index)
        if front:
            gap = front.position[0] - v.position[0]
            if gap < 10:
                r -= 0.5
            elif gap > 20:
                r += 0.3

        # 8. Center-of-lane stabilization
        lane_width = 4.0
        lane_center_y = lane_idx * lane_width
        lateral_pos = v.position[1]
        r -= 0.02 * abs(lateral_pos - lane_center_y)

        # 9. Forward progress reward
        r += 0.07 * (v.speed / 30.0)

        # 10. Reward clipping
        r = float(np.clip(r, -10.0, 10.0))

        return r

    # --------------------------------------------------
    # TERMINATION
    # --------------------------------------------------
    def _is_terminated(self):
        return self.vehicle.crashed

    def _is_truncated(self):
        return (
            self.time >= 120
            or self.vehicle.position[0] > self.config["highway_length"]
        )

    def _reset(self):
        self._create_road()
        self._create_vehicles()


# --------------------------------------------------
# REGISTER ENVIRONMENT
# --------------------------------------------------
try:
    from gymnasium.envs.registration import register
    register(
        id="highway-construction-v0",
        entry_point="multi_stage_env:HighwayConstructionEnv"
    )
    print("highway-construction-v0 registered successfully.")
except Exception as e:
    print(f"Warning registering environment: {e}")
