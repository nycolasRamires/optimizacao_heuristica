import argparse
from rl_training import OmniDroneEnv
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

'''
TODO / ARCHITECTURE NOTES:

The U-Trap Paradox (Zero Memory):
A standard feed-forward MlpPolicy makes decisions based purely on the current observation frame.
When the agent enters a concave obstacle (such as a U-trap) with the target behind the wall,
it can get trapped pushing against the obstacle without temporal awareness to back out.

Mitigation options:
1. Frame Stacking (feed the last 4 frames of LiDAR and motion history).
2. Recurrent Policies (e.g. RecurrentPPO with LSTM from sb3-contrib).
3. Hybrid Global Planner (A* planner supplying sub-waypoints + SAC local collision avoidance).
'''

def make_env(reward_params=None):
    return lambda: OmniDroneEnv(reward_params=reward_params)

def set_learning_rate(model: SAC, new_lr: float):
    """
    Properly updates the learning rate for both SB3's lr_schedule
    and the underlying PyTorch optimizer parameter groups.
    """
    model.learning_rate = new_lr
    model.lr_schedule = lambda _: new_lr
    for param_group in model.actor.optimizer.param_groups:
        param_group["lr"] = new_lr
    for param_group in model.critic.optimizer.param_groups:
        param_group["lr"] = new_lr
    if model.ent_coef_optimizer is not None:
        for param_group in model.ent_coef_optimizer.param_groups:
            param_group["lr"] = new_lr

def train(timesteps=500000, num_envs=16, model_save_name="base", load_model_path=None, learning_rate=None, reward_params=None):
    # Single env health check
    single_env = OmniDroneEnv()
    check_env(single_env)
    single_env.close()

    # Create vectorized environment for high throughput training
    # 16 envs achieves ~2,580 steps/s on RTX 5070 (< 3.5 minutes for 500,000 steps)
    vec_env = DummyVecEnv([make_env(reward_params) for _ in range(num_envs)])

    if load_model_path:
        print(f"Loading pre-trained base model from '{load_model_path}'...")
        model = SAC.load(
            load_model_path,
            env=vec_env,
            device="cuda",
            tensorboard_log="./sac_tensorboard/"
        )
        if learning_rate is not None:
            print(f"Adjusting learning rate to {learning_rate} for warm-start adaptation...")
            set_learning_rate(model, learning_rate)
    else:
        print(f"Initializing fresh SAC policy with {num_envs} vectorized environments...")
        kwargs = {
            "policy": "MlpPolicy",
            "env": vec_env,
            "verbose": 1,
            "tensorboard_log": "./sac_tensorboard/",
            "gamma": 0.999,
            "device": "cuda"
        }
        if learning_rate is not None:
            kwargs["learning_rate"] = learning_rate
        model = SAC(**kwargs)

    print(f"Starting training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, progress_bar=True, reset_num_timesteps=(load_model_path is None))

    # Save trained model
    model.save(model_save_name)
    print(f"Model saved to {model_save_name}.zip")
    vec_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Fine-Tune SAC Drone Agent")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total training timesteps (default: 500,000)")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments (default: 16)")
    parser.add_argument("--save_name", type=str, default="base", help="Model checkpoint save name (default: 'base')")
    parser.add_argument("--load_model", type=str, default=None, help="Path to pre-trained model to warm-start from")
    parser.add_argument("--learning_rate", type=float, default=None, help="Custom learning rate (e.g. 1e-3 for warm-starts)")
    args = parser.parse_args()

    train(
        timesteps=args.timesteps,
        num_envs=args.num_envs,
        model_save_name=args.save_name,
        load_model_path=args.load_model,
        learning_rate=args.learning_rate
    )