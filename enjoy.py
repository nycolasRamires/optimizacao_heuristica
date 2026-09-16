import cv2
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from uav_env import Drone3DEnv
import pybullet as p

CTRL_FREQ  = 42
N_EPISODES = 5
CAM_W, CAM_H = 640, 480
OUT_FILE   = "flights.mp4"

raw_env = Drone3DEnv(gui=False, record=False)
vec_env = DummyVecEnv([lambda: raw_env])
env = VecNormalize.load("vecnormalize_stats.pkl", vec_env)
env.training    = False
env.norm_reward = False

model = SAC.load("navDrone_v1_plus", env=env)

real_env = env.unwrapped.envs[0]
CLIENT   = real_env.CLIENT

proj_matrix = p.computeProjectionMatrixFOV(
    fov=60, aspect=CAM_W / CAM_H, nearVal=0.1, farVal=100.0
)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUT_FILE, fourcc, CTRL_FREQ, (CAM_W, CAM_H))

print(f"Recording {N_EPISODES} episodes → {OUT_FILE}")
for ep in range(1, N_EPISODES + 1):
    obs     = env.reset()
    cam_yaw = 45.0
    print(f"  Episode {ep}/{N_EPISODES}")

    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        state     = real_env._getDroneStateVector(0)
        drone_pos = state[0:3]

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=drone_pos,
            distance=2.5,
            yaw=cam_yaw,
            pitch=-25,
            roll=0,
            upAxisIndex=2
        )

        _, _, rgb_pixels, _, _ = p.getCameraImage(
            width=CAM_W, height=CAM_H,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=CLIENT
        )

        # cv2 expects BGR — PyBullet gives RGBA
        frame_rgb = np.array(rgb_pixels, dtype=np.uint8).reshape(CAM_H, CAM_W, 4)[:, :, :3]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

        cam_yaw += 0.3

        if done[0]:
            print(f"    Ended at step {step}, reward {reward[0]:.1f}")
            # Freeze last frame for 0.5s
            for _ in range(CTRL_FREQ // 2):
                writer.write(frame_bgr)
            break

writer.release()
env.close()
print(f"Done → {OUT_FILE}")