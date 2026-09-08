from rl_training import OmniDroneEnv
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

'''
TODO 
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


# get and check env
env = OmniDroneEnv()
check_env(env)

# (Soft Actor-Critic)
# MlpPolicy -> simple networks
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_tensorboard/", gamma=0.999)

print("Training")
model.learn(total_timesteps=1750000, progress_bar=True)

# save trained model
model.save("sac_drone_v7")