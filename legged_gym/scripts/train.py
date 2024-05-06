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

import numpy as np
import os
#from datetime import datetime

#import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import subprocess
import sys

import time
import threading

rollout_path = os.path.join(os.path.dirname(__file__), 'rollout.py')
def register_task(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg, log_dir = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    return ppo_runner, train_cfg, log_dir

def train(args, ppo_runner, train_cfg):
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


def run_subprocess(log_dir, args):
    process = subprocess.Popen(["python", rollout_path, "--log_dir=%s"%log_dir, *args]) #stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    return process
    # pass forward args from training
    if debug:
        stdout, stderr = process.communicate()
        print("Subprocess output:", stdout.decode())
        if process.returncode != 0:
            print("Error:", stderr.decode())

 


if __name__ == '__main__':
    args = get_args()

    ppo_runner, train_cfg, log_dir = register_task(args)

    process = run_subprocess(log_dir, sys.argv[1:])
    train(args, ppo_runner, train_cfg)

    # Start the subprocess in a separate thread
#    subprocess_thread = threading.Thread(target=run_subprocess, args=(log_dir, sys.argv[1:]))
#    subprocess_thread.start()
#    subprocess_thread.join()
#    time.sleep(10)
    process.terminate()
    exit_code = process.wait()

    # Check the exit code
    if exit_code == 0:
        print("Subprocess completed successfully.")
    else:
        print("Subprocess failed with exit code:", exit_code)




