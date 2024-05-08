# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import signal
import threading


stop_event = threading.Event()

def sigterm_handler(signal, frame):
    global stop_event
    stop_event.set()
    print("Received SIGTERM. No longer processing new models.")

# Need to register early to catch at start time
signal.signal(signal.SIGTERM, sigterm_handler)

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

#import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, task_registry, Logger

import numpy as np
import torch
import queue


import sys
import time
import tempfile
import shutil

from contextlib import contextmanager
from isaacgym import gymapi

SIM_FPS = 30
VIDEO_FPS = float(10.0)
RECORD_FREQ = SIM_FPS//VIDEO_FPS
RECORD_S = 20
TERM_ITR = RECORD_FREQ * RECORD_S
#Experiment, see note below
RECORD_MULTI = False

_sentinel = object()

@contextmanager
def make_tmp_dir():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

def rollout(q, log_dir, args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    set_env_params(env_cfg)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if not RECORD_MULTI:
        camera_handle = env.gym.create_camera_sensor(env.envs[0], gymapi.CameraProperties())
        print(camera_handle)
        env.gym.set_camera_location(camera_handle, env.envs[0], gymapi.Vec3(*env_cfg.viewer.pos), gymapi.Vec3(1.,1.,0))
    else:
        env.gym.viewer_camera_look_at(env.viewer, None, gymapi.Vec3(10,10,10), gymapi.Vec3(0,0,0))
    # load policy
    train_cfg.runner.resume = True
 
    #blocking process
    while (file := q.get()) is not _sentinel:
        train_cfg.runner.model_path = file
        file_basename = os.path.basename(file)
        ppo_runner, _, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
        policy = ppo_runner.get_inference_policy(device=env.device)
        rollout_single(env, camera_handle, policy, log_dir, env_cfg, file_basename)
        env.reset()
        q.task_done()

def rollout_single(env, camera_handle, policy, log_dir, env_cfg, filename, record=True, move_camera=False):
    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 500 # number of steps before plotting states
    stop_state_log = -1 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)

    with make_tmp_dir() as frames_dir:
        obs = env.get_observations()
        #for i in range(int(env.max_episode_length)):
        for i in range(int(TERM_ITR)):
 
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
            if record:
                if i % RECORD_FREQ == 0:
                    frame_name = "frame%d.png" % (i)
                    if RECORD_MULTI: 
                        # Experimental, Isaacgym requires the viewer to be drawn for the frames to be writable. This results in significant overhead and possibly requires a screen handle (won't work in headless). It provides the benefit of viewing multiple environments instead of just one. An alternative is stacking frames from multiple cameras.
                        env.gym.draw_viewer(env.viewer, env.sim)
                        env.gym.write_viewer_image_to_file(env.viewer, os.path.join(frames_dir, frame_name))
                    else:
                        env.gym.render_all_camera_sensors(env.sim)
                        env.gym.write_camera_image_to_file(env.sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR, os.path.join(frames_dir, frame_name))

            if move_camera:
                #TODO update this for env camera rendering
                camera_position += camera_vel * env.dt
                env.set_camera(camera_position, camera_position + camera_direction)

            if i < stop_state_log:
                logger.log_states(
                    {
                        'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                        'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                        'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                        'dof_torque': env.torques[robot_index, joint_index].item(),
                        'command_x': env.commands[robot_index, 0].item(),
                        'command_y': env.commands[robot_index, 1].item(),
                        'command_yaw': env.commands[robot_index, 2].item(),
                        'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                        'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                        'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                        'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                        'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    }
                )

            elif i==stop_state_log:
                logger.plot_states()

            if  0 < i < stop_rew_log:
                if infos["episode"]:
                    num_episodes = torch.sum(env.reset_buf).item()
                    if num_episodes>0:
                        logger.log_rewards(infos["episode"], num_episodes)
            elif i==stop_rew_log:
                logger.print_rewards()

        if record:
            create_video_from_images(frames_dir, os.path.join(log_dir, filename + '.mp4'))

def set_env_params(env_cfg):
    # override some parameters for testing
    env_cfg.env.num_envs = 1 if not RECORD_MULTI else 4 
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False



import cv2
def get_creation_time(file_path):
    return os.path.getctime(file_path)
def sort_files_by_created_time(directory):
    # Get a list of all files in the directory
    file_list = os.listdir(directory)

    # Create a list of tuples, each containing the file name and its creation time
    file_with_creation_times = [(file, get_creation_time(os.path.join(directory, file))) for file in file_list]

    # Sort the list based on the creation time (second element of the tuple)
    sorted_files = sorted(file_with_creation_times, key=lambda x: x[1])

    # Extract only the file names from the sorted list
    sorted_file_names = [file[0] for file in sorted_files]

    return sorted_file_names

def create_video_from_images(image_dir, output_path, frame_rate=VIDEO_FPS):
    images = sort_files_by_created_time(image_dir)

    frame = cv2.imread(os.path.join(image_dir, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    for img in images:
        video.write(cv2.imread(os.path.join(image_dir,img)))

    cv2.destroyAllWindows()
    video.release()

def get_files(dir_path, exclude=[], file_extension='.pt'):
    files = os.listdir(dir_path)
    files = [file for file in files if os.path.isfile(os.path.join(dir_path, file)) and file not in exclude and file.endswith(file_extension)]
    return files

def process_files(dir_path, files, q):
    for file in files:
        print("Recording Thread: Processing", file)
        file_path = os.path.join(dir_path, file)
        q.put(file_path)

def process_files_sub(dir_path, args):
    q = queue.Queue()
    sim_thread = threading.Thread(target=rollout, args=(q, dir_path, args))
    sim_thread.start()

    processed = []

    while not os.path.exists(dir_path):
        time.sleep(5) 
        print("Recording Thread: Waiting for log directory creation")

    while True:
        files = get_files(dir_path, processed)
        process_files(dir_path, files, q)
        processed.extend(files)

        if stop_event.is_set():
            break;
        time.sleep(10)

    files = get_files(dir_path, processed)
    process_files(dir_path, files, q)

    q.put(_sentinel)

    sim_thread.join()

def run_playback_recording_thread(args, dir_path):
    # Create a separate thread for file processing
    file_processing_thread = threading.Thread(target=process_files, args=(dir_path, args), daemon=True)
    file_processing_thread.start()
    return file_processing_thread

if __name__ == '__main__':
    args = get_args() 
    process_files_sub(args.log_dir, args)
