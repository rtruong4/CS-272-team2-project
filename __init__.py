from gymnasium.envs.registration import register

# Register the new single highway environment
register(
    id="highway-construction-v0",
    entry_point="multi_stage_env:HighwayConstructionEnv"
)

print(" highway-construction-v0 registered successfully.")
