import numpy as np
from highway_env.envs.highway_env import HighwayEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane
from highway_env.vehicle.behavior import IDMVehicle


class HighwayConstructionEnv(HighwayEnv):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update({
            "lanes_count": 4,
            "vehicles_count": 25,
            "duration": 120,
            "screen_width": 1200,
            "screen_height": 400,
            "scaling": 5.5,
            "collision_reward": -300,
            "highway_length": 1000,
            "centering_position": [0.3, 0.5],
            "simulation_frequency": 10,
            "policy_frequency": 5,
        })
        return cfg

    # Road Layout

    def _create_road(self):
        net = RoadNetwork()
        lane_w = 4.0
        lanes = self.config["lanes_count"]
        L = self.config["highway_length"]

        # --- Main 4-lane highway ---
        for i in range(lanes):
            origin = np.array([0, i * lane_w])
            end = np.array([L, i * lane_w])
            net.add_lane("a", "b", StraightLane(origin, end, width=lane_w))

        # --- Entrance ramp ---
        ramp_start = np.array([L * 0.05, -lane_w * 2])
        ramp_end = np.array([L * 0.2, lane_w])
        net.add_lane("entrance", "a",
                     StraightLane(ramp_start, ramp_end, width=lane_w))

        # --- Exit ramp ---
        exit_start = np.array([L * 0.8, 0])
        exit_end = np.array([L * 0.9, -lane_w * 2])
        net.add_lane("b", "exit",
                     StraightLane(exit_start, exit_end, width=lane_w))

        self.road = Road(net, np_random=self.np_random)

    # Vehicle Spawning

    def _create_vehicles(self):
        """Spawn ego on highway + traffic + slow cars + construction cones."""
        from highway_env.vehicle.controller import ControlledVehicle

        # Conversion factor: mph -> m/s
        MPH_TO_MS = 0.44704

        #  Ego (green) car: prefers 60–70 mph (~26.8–31.3 m/s) ---
        ego_lane = self.road.network.get_lane(("a", "b", 1))
        start_speed = np.random.uniform(26.8, 31.3)
        ego_vehicle = ControlledVehicle(self.road, ego_lane.position(50, 0), speed=start_speed)
        self.vehicle = ego_vehicle
        self.road.vehicles.append(ego_vehicle)

        # Regular traffic: 60–100 mph (26.8–44.7 m/s)
        for _ in range(self.config["vehicles_count"]):
            lane = self.np_random.choice(self.road.network.lanes_list())
            long = self.np_random.uniform(0, lane.length)
            x, y = lane.position(long, 0)
            speed = self.np_random.uniform(26.8, 44.7)
            v = IDMVehicle(self.road, [x, y], speed=speed)
            self.road.vehicles.append(v)

        #  Slow cars: 35 mph (15.6 m/s)
        for _ in range(3):
            lane = self.np_random.choice(self.road.network.lanes_list())
            long = self.np_random.uniform(200, 800)
            x, y = lane.position(long, 0)
            v = IDMVehicle(self.road, [x, y], speed=15.6)
            self.road.vehicles.append(v)

        #  Construction cones (static)
        construction_lane_index = int(self.np_random.choice([0, self.config["lanes_count"] - 1]))
        construction_lane = self.road.network.get_lane(("a", "b", construction_lane_index))
        for offset in [400, 410, 420, 430, 440, 450]:
            x, y = construction_lane.position(offset, 0)
            cone = IDMVehicle(self.road, [x, y], speed=0, target_speed=0)
            cone.color = (255, 120, 0)  # orange cones
            self.road.vehicles.append(cone)

    # Reward Function (in mph)

    def _reward(self, action):
        v = self.vehicle
        r = 0.0
        mph = v.speed / 0.44704  # convert m/s -> mph

        # Crash penalty
        if v.crashed:
            return self.config["collision_reward"]

        # Speed reward: ideal range 60–70 mph
        lower_bound = 60
        upper_bound = 70
        optimal_speed = 65
        if lower_bound <= mph <= upper_bound:
            if mph == optimal_speed:
                r += 2.0
            else:
                r += 1.0 - ((mph - optimal_speed) / 5)**2
        else:
            r += -1.0 * ((mph - optimal_speed) / 5)**2

        #  Lane change logic
        if action in [3, 4]:
            r -= 1.0

        # --- Construction zone penalty ---
        if 400 < v.position[0] < 460:
            lane_idx = v.lane_index[2] if hasattr(v, "lane_index") else 0
            if lane_idx == 0 or lane_idx == self.config["lanes_count"] - 1:
                r -= 5.0

        r += 0.03 * v.speed / 30

        return r

    # Termination

    def _is_terminated(self):
        return self.vehicle.crashed

    def _is_truncated(self):
        return self.time >= 120 or self.vehicle.position[0] > self.config["highway_length"]

    def _reset(self):
        self._create_road()
        self._create_vehicles()


# Register Environment
try:
    from gymnasium.envs.registration import register

    register(
        id="highway-construction-v0",
        entry_point="multi_stage_env:HighwayConstructionEnv"
    )
    print(" highway-construction-v0 registered successfully.")
except Exception as e:
    print(f"Warning registering environment: {e}")
