
'''
TEst teste4
IDEA FROM AI

The U-Trap Paradox (Zero Memory)
Here is the catch with MlpPolicy: it has absolutely zero memory.

A standard MLP is a feed-forward network. It only makes decisions based on the exact current frame. It has no concept of the past.
If you throw this agent into the U-trap without the A* global planner guiding it, it will fly to the bottom of the "U", see the target directly through the wall, 
and push its face against the concrete forever. Because it cannot remember how it got into the U-trap, it cannot deduce that it needs to turn around and fly away from the target to escape.

To solve complex mazes purely end-to-end without A*, an agent needs short-term memory to realize "I have been stuck here for 3 seconds." 
Stable Baselines 3's SAC does not natively support Recurrent Neural Networks (LSTMs). If you want to solve the U-trap later, you will either need to:

Switch to RecurrentPPO (which has LSTM memory).

Implement Frame Stacking (feeding the agent the last 4 frames of LiDAR so it can infer its own movement history).

Bring back the A* planner.
'''

import os
from uav_env import Drone3DEnv
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

MODEL_PATH = "navDrone_v1"
STATS_PATH = "vecnormalize_stats.pkl"
restart_training = False
TIMESTEPS  = 3_500_000

check_env(Drone3DEnv())

env = Monitor(Drone3DEnv())
env = DummyVecEnv([lambda: env])

resuming = os.path.isfile(MODEL_PATH + ".zip") and os.path.isfile(STATS_PATH)

# will resume training a model if MODEL_PATH is the name os an already existing false and restart_training==false
if resuming and not restart_training:
    print(f"Resuming from {MODEL_PATH}.zip + {STATS_PATH}")
    env = VecNormalize.load(STATS_PATH, env)
    env.training    = True    # re-enable stat updates for continued training
    env.norm_reward = False
    model = SAC.load(MODEL_PATH, env=env)
    # Restore the replay buffer if it was saved
    # gives the optimizer a start instead of exploring from scratch.
    if os.path.isfile(MODEL_PATH + "_replay_buffer.pkl"):
        print("  Loading replay buffer...")
        model.load_replay_buffer(MODEL_PATH + "_replay_buffer.pkl")
else:
    print("No checkpoint found — starting fresh.")
    env = VecNormalize(env, norm_obs=True, norm_obs_keys=["state"], norm_reward=False, clip_obs=10.0)
    model = SAC(
        "MultiInputPolicy",
        env,
        buffer_size=100_000,
        replay_buffer_kwargs={"handle_timeout_termination": False},
        batch_size=128,
        verbose=1,
        tensorboard_log="./3d_tensorboard/",
    )


model.learn(
    total_timesteps=TIMESTEPS,
    reset_num_timesteps=not resuming,   # keeps the TensorBoard x-axis continuous
    progress_bar=True,
)

# saving model
model.save(MODEL_PATH)
env.save(STATS_PATH)
model.save_replay_buffer(MODEL_PATH + "_replay_buffer.pkl")
print(f"Saved → {MODEL_PATH}.zip, {STATS_PATH}, replay buffer")
