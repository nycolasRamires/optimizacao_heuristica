import os
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from uav_env import Drone3DEnv

# Base checkpoint to warm-start from
BASE_MODEL_PATH = "navDrone_v1"
BASE_STATS_PATH = "vecnormalize_stats.pkl"

# Target checkpoint to train and save into
MODEL_PATH = "navDrone_v2"
STATS_PATH = "vecnormalize_stats_v2.pkl"
RESTART_TRAINING = False
TIMESTEPS = 10_000_000

# High-throughput vectorization settings (12 workers across CPU cores)
N_ENVS = 12
BATCH_SIZE = 256
BUFFER_SIZE = 200_000
TRAIN_FREQ = (4, "step")
GRADIENT_STEPS = 4

def make_env():
    # Factory returning monitored 3D drone environment for each worker process
    return Monitor(Drone3DEnv(gui=False))

if __name__ == "__main__":
    # Validate environment compliance with Gymnasium interface before spawning workers
    check_env(Drone3DEnv())

    # Parallel multiprocessing environment to distribute PyBullet rendering across CPU cores
    vec_env = SubprocVecEnv([make_env for _ in range(N_ENVS)])

    resuming_v2 = os.path.isfile(MODEL_PATH + ".zip") and os.path.isfile(STATS_PATH)
    warm_start_v1 = os.path.isfile(BASE_MODEL_PATH + ".zip") and os.path.isfile(BASE_STATS_PATH)

    # Resume v2 if already in progress, else warm-start from v1, else start fresh
    if resuming_v2 and not RESTART_TRAINING:
        print(f"Resuming {MODEL_PATH} from {MODEL_PATH}.zip + {STATS_PATH}")
        env = VecNormalize.load(STATS_PATH, vec_env)
        env.training = True
        env.norm_reward = False
        model = SAC.load(MODEL_PATH, env=env)
        if os.path.isfile(MODEL_PATH + "_replay_buffer.pkl"):
            print("  Loading v2 replay buffer...")
            model.load_replay_buffer(MODEL_PATH + "_replay_buffer.pkl")
    elif warm_start_v1 and not RESTART_TRAINING:
        print(f"Warm-starting {MODEL_PATH} from base {BASE_MODEL_PATH}.zip + {BASE_STATS_PATH}")
        env = VecNormalize.load(BASE_STATS_PATH, vec_env)
        env.training = True
        env.norm_reward = False
        model = SAC.load(BASE_MODEL_PATH, env=env)
    else:
        print(f"Starting fresh training with {N_ENVS} parallel environments.")
        env = VecNormalize(vec_env, norm_obs=True, norm_obs_keys=["state"], norm_reward=False, clip_obs=10.0)
        model = SAC(
            "MultiInputPolicy",
            env,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            train_freq=TRAIN_FREQ,
            gradient_steps=GRADIENT_STEPS,
            replay_buffer_kwargs={"handle_timeout_termination": False},
            verbose=0,
            tensorboard_log="./3d_tensorboard/",
        )

    # Train SAC policy with continuous TensorBoard tracking
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=not resuming_v2,
        progress_bar=True,
    )

    # Save model weights, normalization statistics, and replay buffer
    model.save(MODEL_PATH)
    env.save(STATS_PATH)
    model.save_replay_buffer(MODEL_PATH + "_replay_buffer.pkl")
    print(f"Saved -> {MODEL_PATH}.zip, {STATS_PATH}, replay buffer")
    env.close()
