import os
import argparse
import time
import cv2
import imageio
import numpy as np
import pybullet as p
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from uav_env import Drone3DEnv

default_model = "navDrone_v2" if os.path.isfile("navDrone_v2.zip") else "navDrone_v1"
default_stats = "vecnormalize_stats_v2.pkl" if os.path.isfile("vecnormalize_stats_v2.pkl") else "vecnormalize_stats.pkl"

# Command line interface configuration
parser = argparse.ArgumentParser(description="Visualize 3D quadrotor policy in interactive GUI or cinematic HD video.")
parser.add_argument("--video", action="store_true", help="Record HD video instead of opening interactive GUI")
parser.add_argument("--model", type=str, default=default_model, help="Model checkpoint name (without .zip)")
parser.add_argument("--stats", type=str, default=default_stats, help="VecNormalize stats path")
parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
parser.add_argument("--width", type=int, default=1280, help="Video width for --video mode")
parser.add_argument("--height", type=int, default=720, help="Video height for --video mode")
parser.add_argument("--fps", type=int, default=60, help="Video playback framerate")
parser.add_argument("--slowmo", type=float, default=0.5, help="Playback speed multiplier (0.5 = 2x slow-motion for smooth flight)")
parser.add_argument("--out", type=str, default="flights.mp4", help="Output video file path")
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier in GUI mode")
args = parser.parse_args()

CTRL_FREQ = 40
use_gui = not args.video

# Load SAC policy on CPU before initializing PyBullet GUI to prevent OpenGL driver collision
model = SAC.load(args.model, custom_objects={'n_envs': 1}, device='cpu')

# Initialize environment with GUI or headless video rendering
raw_env = Drone3DEnv(gui=use_gui, record=False)
vec_env = DummyVecEnv([lambda: raw_env])
env = VecNormalize.load(args.stats, vec_env)
env.training = False
env.norm_reward = False

# Attach evaluation environment to model
model.set_env(env)
real_env = env.unwrapped.envs[0]
CLIENT = real_env.CLIENT

# Configure interactive 3D visualizer settings
if use_gui:
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=CLIENT)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=CLIENT)
    print(f"\n[GUI Mode] Visualizing '{args.model}' for {args.episodes} episodes in real time...")
    print("Mouse controls: Left-click + drag to orbit, right-click to pan, scroll to zoom.\n")
else:
    # Setup high-quality H.264 video writer via imageio-ffmpeg
    writer = imageio.get_writer(args.out, fps=args.fps, codec='libx264', quality=8, pixelformat='yuv420p')
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=args.width / args.height, nearVal=0.1, farVal=100.0
    )
    print(f"\n[Video Mode] Recording {args.episodes} episodes to '{args.out}' ({args.width}x{args.height} @ {args.fps} FPS, slowmo={args.slowmo}x)...")

# Main evaluation loop across episodes
for ep in range(1, args.episodes + 1):
    obs = env.reset()
    target_pos = real_env.goal_pos.copy()
    state = real_env._getDroneStateVector(0)
    drone_pos = state[0:3].copy()
    smooth_pos = drone_pos.copy()
    cam_yaw = 45.0
    ep_return = 0.0

    for step in range(500):
        pre_pos = state[0:3].copy()
        pre_dist = float(np.linalg.norm(target_pos - pre_pos))

        # Predict deterministic action and step simulation
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        rew = float(reward[0])
        ep_return += rew

        state = real_env._getDroneStateVector(0)
        curr_drone_pos = pre_pos if done[0] else state[0:3].copy()

        # Smooth camera target to eliminate quadrotor PID micro-vibrations
        prev_smooth = smooth_pos.copy()
        smooth_pos = 0.85 * smooth_pos + 0.15 * curr_drone_pos

        if use_gui:
            # Update interactive viewer camera and maintain real-time pacing
            cam_yaw += 0.25
            p.resetDebugVisualizerCamera(
                cameraDistance=2.5,
                cameraYaw=cam_yaw,
                cameraPitch=-25.0,
                cameraTargetPosition=smooth_pos,
                physicsClientId=CLIENT
            )
            time.sleep((1.0 / CTRL_FREQ) / args.speed)
        else:
            # Sub-frame interpolation for buttery-smooth 60 FPS video
            sub_frames = max(1, int(round(args.fps / (CTRL_FREQ * args.slowmo))))
            for sub in range(sub_frames):
                alpha = (sub + 1) / float(sub_frames)
                interp_pos = (1.0 - alpha) * prev_smooth + alpha * smooth_pos
                sub_yaw = cam_yaw + alpha * 0.25

                view_matrix = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=interp_pos,
                    distance=2.5,
                    yaw=sub_yaw,
                    pitch=-25.0,
                    roll=0,
                    upAxisIndex=2
                )
                _, _, rgb_pixels, _, _ = p.getCameraImage(
                    width=args.width, height=args.height,
                    viewMatrix=view_matrix, projectionMatrix=proj_matrix,
                    flags=p.ER_NO_SEGMENTATION_MASK,
                    renderer=p.ER_TINY_RENDERER,
                    physicsClientId=CLIENT
                )
                frame_rgb = np.array(rgb_pixels, dtype=np.uint8).reshape(args.height, args.width, 4)[:, :, :3]

                # Draw translucent HUD banner and flight telemetry
                overlay = frame_rgb.copy()
                cv2.rectangle(overlay, (0, 0), (args.width, 44), (20, 20, 20), -1)
                frame_rgb = cv2.addWeighted(overlay, 0.6, frame_rgb, 0.4, 0)

                dist_val = pre_dist if done[0] else float(np.linalg.norm(target_pos - curr_drone_pos))
                speed_val = float(np.linalg.norm(state[10:13]))
                hud_text = f"Ep {ep}/{args.episodes} | Step {step+1:03d} | Goal Dist: {dist_val:.2f}m | Speed: {speed_val:.1f}m/s | Return: {ep_return:.1f}"
                cv2.putText(frame_rgb, hud_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                writer.append_data(frame_rgb)

            cam_yaw += 0.25

        if done[0]:
            # Accurate termination detection avoiding DummyVecEnv auto-reset overwrite
            is_goal = rew > 50.0 or info[0].get("is_success", False) or pre_dist < 0.25
            is_timeout = step >= 499 or info[0].get("timeout", False)
            is_coll = (info[0].get("collision", False) or not is_goal) and not is_timeout
            status = "GOAL REACHED" if is_goal else ("COLLISION" if is_coll else "TIMEOUT")
            final_dist = pre_dist if is_goal else float(np.linalg.norm(target_pos - curr_drone_pos))
            print(f"  Episode {ep:02d}/{args.episodes}: Steps={step+1:03d}, Return={ep_return:6.1f}, Final Dist={final_dist:.2f}m -> [{status}]")

            if not use_gui:
                # Overlay prominent outcome banner and hold for victory freeze
                color = (50, 220, 50) if is_goal else ((220, 50, 50) if is_coll else (255, 180, 50))
                banner_text = f"STATUS: {status} in {step+1} STEPS!" if is_goal else f"STATUS: {status}"
                cv2.putText(frame_rgb, banner_text, (16, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
                for _ in range(int(args.fps * 1.5)):
                    writer.append_data(frame_rgb)
            break

# Cleanup simulation and video resources
if not use_gui:
    writer.close()
    print(f"\nDone -> High-definition video successfully exported to '{args.out}'")
env.close()