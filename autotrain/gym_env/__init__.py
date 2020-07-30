from gym.envs.registration import register

register(
    id='AutoTrain-v0',
    entry_point='autotrain.gym_env.envs:AutoTrainEnvironment',
)