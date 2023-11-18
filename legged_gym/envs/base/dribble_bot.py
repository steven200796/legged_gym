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
from legged_gym import LEGGED_GYM_RESOURCES_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class DribbleBot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            # need these if resetting on some rigid body position
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[self.robot_actor_idxs, 0:3]
        self.base_quat[:] = self.root_states[self.robot_actor_idxs, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[self.robot_actor_idxs, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[self.robot_actor_idxs, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.projected_gravity_accel[:] = quat_rotate_inverse(self.base_quat, self.gravity_accel)

        self.ball_pos = self.root_states[self.ball_actor_idxs, 0:3]
        self.ball_quat = self.root_states[self.ball_actor_idxs, 3:7]
#        self.ball_lin_vel = quat_rotate_inverse(self.ball_quat, self.root_states[self.ball_actor_idxs, 7:10])
        self.ball_lin_vel = self.root_states[self.ball_actor_idxs, 7:10]

        self.ball_pos_robot_frame = quat_rotate_inverse(self.base_quat, self.ball_pos - self.base_pos)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
#        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1) 
 #       print(self.asset_body_names['nao'][(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.)])
        # terminate if body is below threshold
        self.reset_buf = (torch.squeeze(self.rigid_body_pos[:, self.termination_height_indices, 2], 1) < 0.27).view(-1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def expand_env_ids(self, env_ids):
        #todo make this seamless in tensor access
        actor_ids = []
        for env_id in env_ids:
            actor_ids.extend(range(env_id * self.num_actors_per_env, (env_id + 1)* self.num_actors_per_env))
        return torch.Tensor(actor_ids).to(device=self.device,dtype=torch.long)


    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

#        env_ids = self.expand_env_ids(env_ids)
 
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    ((self.accel_tensor[:,:3] / self.masses) + self.projected_gravity_accel) * 0.1,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.ball_pos_robot_frame
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id // self.num_actors_per_env]

        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        self.masses[env_id] = sum([x.mass for x in props])
        return props
 
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:            
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        actor_ids = env_ids.detach().clone()
        self.dof_pos[actor_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(actor_ids), self.num_dof), device=self.device)
        self.dof_vel[actor_ids] = 0.

        actor_ids_int32 = self.robot_actor_idxs[actor_ids].to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(actor_ids_int32), len(actor_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        # convert actor ids to environment ids
        actor_ids_relative = env_ids.detach().clone()
        actor_ids = self.robot_actor_idxs[actor_ids_relative]
        env_ids = actor_ids_relative // self.num_actors_per_env

        self.root_states[actor_ids] = self.base_init_state
        self.root_states[actor_ids, :3] += self.env_origins[env_ids] + self.env_actor_offsets[actor_ids_relative]
        self.root_states[actor_ids, 3:7] = self.env_actor_rotations[actor_ids_relative]
        if self.custom_origins:
            self.root_states[actor_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        # base velocities
        self.root_states[actor_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        actor_ids_int32 = actor_ids.to(dtype=torch.int32)


        ball_idxs = self.ball_actor_idxs[env_ids]
        self.root_states[ball_idxs, :3] = self.env_origins[env_ids]
        #todo flag randomization of ball here
        #self.root_states[ball_idxs, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
        self.root_states[ball_idxs, :1] += self.cfg.domain_rand.max_ball_distance
        self.root_states[ball_idxs, 2] = 0.08
        self.root_states[ball_idxs, 7:13] = 0.
        actor_ids_int32 = torch.cat((actor_ids_int32, ball_idxs.to(dtype=torch.int32)), dim=-1)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(actor_ids_int32), len(actor_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[self.robot_actor_idxs, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs * self.num_actors_per_env, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    
    def _get_noise_scale_vec(self, cfg, num_commands):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level


        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands


        i = 12
        noise_vec[i:i+num_commands] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        i += num_commands
        noise_vec[i:i+num_commands] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        i += num_commands
        noise_vec[i:i+num_commands] = 0. # previous actions
        i += num_commands
        noise_vec[i:i+6] = 0
        if self.cfg.terrain.measure_heights:
            noise_vec[i:i+187] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        force_states = self.gym.acquire_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)


        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.body_states = gymtorch.wrap_tensor(rigid_body_states)
        self.dof_pos = self.dof_state.view(self.num_envs * self.num_actors_per_env, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs * self.num_actors_per_env, self.num_dof, 2)[..., 1]
        #todo, fill in -1 with asset rigid body count

#        self.rigid_body_pos = self.body_states.view(self.num_envs, self.num_bodies_per_env, 13)[:, :self.num_agent_bodies_per_env, 0:3], (self.num_envs * self.num_actors_per_env, -1, 3),  #.view(self.num_envs * self.num_actors_per_env, -1, 3)
        self.rigid_body_pos = self.body_states.view(self.num_envs, self.num_bodies_per_env, 13)[:, :self.num_agent_bodies_per_env, 0:3]
        self.rigid_body_vel = self.body_states.view(self.num_envs, self.num_bodies_per_env, 13)[:, :self.num_agent_bodies_per_env, 7:10]#.view(self.num_envs * self.num_actors_per_env, -1, 3)
        self.accel_tensor = gymtorch.wrap_tensor(force_states)
        self.base_pos = self.root_states[self.robot_actor_idxs, 0:3]
        self.base_quat = self.root_states[self.robot_actor_idxs, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, self.num_bodies_per_env, 3)[:, :self.num_agent_bodies_per_env, :]  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg, self.cfg.commands.num_commands)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs * self.num_actors_per_env, 1))
        self.gravity_accel = to_torch(get_axis_params(9.81, self.up_axis_idx), device=self.device).repeat((self.num_envs * self.num_actors_per_env, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs * self.num_actors_per_env, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs * self.num_actors_per_env, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs * self.num_actors_per_env, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs * self.num_actors_per_env, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs * self.num_actors_per_env, self.feet_indices.shape[1], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs * self.num_actors_per_env, self.feet_indices.shape[1], dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[self.robot_actor_idxs, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[self.robot_actor_idxs, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.projected_gravity_accel = quat_rotate_inverse(self.base_quat, self.gravity_accel)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0


        self.ball_pos = self.root_states[self.ball_actor_idxs, 0:3]
        self.ball_quat = self.root_states[self.ball_actor_idxs, 3:7]
#       self.ball_lin_vel = quat_rotate_inverse(self.ball_quat, self.root_states[self.ball_actor_idxs, 7:10])
        self.ball_lin_vel = self.root_states[self.ball_actor_idxs, 7:10]
        self.ball_pos_robot_frame = quat_rotate_inverse(self.base_quat, self.ball_pos - self.base_pos)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.init_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        if 'joint' in self.cfg.control.stiffness:  
            self.p_gains[:] = self.cfg.control.stiffness['joint']
        if 'joint' in self.cfg.control.damping:  
            self.d_gains[:] = self.cfg.control.damping['joint']
        for i in range(self.asset_num_dof[self.asset_name]):
            name = self.asset_dof_names[self.asset_name][i]
            angle = 0

            self.default_dof_pos[i] = 0.5 * (self.dof_pos_limits[i, 1] + self.dof_pos_limits[i, 0])
            self.init_dof_pos[i] = self.default_dof_pos[i]
            if not self.cfg.init_state.use_halfway:
                self.init_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
            for dof_tag in self.cfg.control.stiffness.keys():
                if dof_tag in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_tag]
            for dof_tag in self.cfg.control.damping.keys():
                if dof_tag in name:
                    self.d_gains[i] = self.cfg.control.damping[dof_tag]
        print(self.p_gains, self.d_gains)
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs * self.num_actors_per_env, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _load_assets(self):
        self.asset_num_dof = {}
        self.asset_num_bodies = {}
        self.asset_dof_props = {}
        self.asset_rigid_shape_props = {}
        self.asset_body_names = {}
        self.asset_dof_names = {}

        # todo Multi asset branch, assets referenced in config which in turn reference object files with asset_options defined
        count = 2
        if False:
            assets = self.cfg.assets
 
        self.assets = {}
        for name, asset_path in self.cfg.asset.asset_paths.items():
            #asset = asset_module._load_asset()
            asset = self._load_asset(asset_path)
            # todo read imu rigid body from config
            self._add_imu(asset)
            self.assets[name] = asset

            self.asset_num_dof[name] = self.gym.get_asset_dof_count(asset)
            self.asset_num_bodies[name] = self.gym.get_asset_rigid_body_count(asset)
            self.asset_dof_props[name] = self.gym.get_asset_dof_properties(asset)
            self.asset_rigid_shape_props[name] = self.gym.get_asset_rigid_shape_properties(asset)
            self.asset_body_names[name] = self.gym.get_asset_rigid_body_names(asset)
            self.asset_dof_names[name] = self.gym.get_asset_dof_names(asset)



    def _load_asset(self, asset_path):
        #class out options for multiple assets
        asset_options = parse_cfg_asset_options(self.cfg)
        asset = self.gym.load_asset(self.sim, LEGGED_GYM_RESOURCES_DIR, asset_path, asset_options)
        return asset

    def _add_imu(self, asset, imu_rigid_body_name="ImuTorsoAccelerometer_frame"):
        body_idx = self.gym.find_asset_rigid_body_index(asset, imu_rigid_body_name)
        sensor_pose = gymapi.Transform()
        self.gym.create_asset_force_sensor(asset, body_idx, sensor_pose)

    def _setup_penalties_and_terminations(self):
        feet_names = [s for s in self.asset_body_names[self.asset_name] if self.cfg.asset.feet_names[self.asset_name] in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.asset_body_names[self.asset_name] if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.asset_body_names[self.asset_name] if name in s])
        
        # All the following assume all envs are created with the same actors in order
        self.feet_indices = torch.zeros(len(self.robot_actor_handles), len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names) * len(self.robot_actor_handles), dtype=torch.long, device=self.device, requires_grad=False)
        self.termination_contact_indices = torch.zeros(len(termination_contact_names) * len(self.robot_actor_handles), dtype=torch.long, device=self.device, requires_grad=False)
        self.termination_height_indices = torch.zeros(len(self.robot_actor_handles), dtype=torch.long, device=self.device, requires_grad=False)

        num_feet = len(feet_names)
        num_pen_cont = len(penalized_contact_names)
        num_term_cont = len(termination_contact_names) 
        num_term_height = 1 

        for i in range(len(self.robot_actor_handles)):
            for j in range(num_feet):
                self.feet_indices[i, j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_actor_handles[i], feet_names[j])
            for j in range(num_pen_cont):
                self.penalised_contact_indices[num_pen_cont * i + j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_actor_handles[i], penalized_contact_names[j])
            for j in range(num_term_cont):
                self.termination_contact_indices[num_term_cont * i + j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_actor_handles[i], termination_contact_names[j])
            for j in range(num_term_height): 
                # todo read this from config
                self.termination_height_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_actor_handles[i], "Head")

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        self._load_assets()
        # todo load this from config
        positions = self.cfg.asset.positions
        rotations = self.cfg.asset.rotations

        #todo default to first asset as heterogeneous agents not supported
        self.asset_name = list(positions.keys())[0]

        self.num_dof = self.asset_num_dof[self.asset_name] #sum([self.asset_num_dof[asset_name] * len(positions[asset_name]) for asset_name in self.asset_names])
        self.num_agent_bodies_per_env = sum([self.asset_num_bodies[asset_name] * len(positions[asset_name]) for asset_name in positions.keys()])
        self.num_bodies_per_env = self.num_agent_bodies_per_env
#        self.num_actuated_dof = sum([asset for asset inall_assets])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)

        self.robot_actor_handles = []
        self.object_actor_handles = []
        self.ball_actor_handles = []

        self.robot_actor_idxs = []
        self.object_actor_idxs = []
        self.ball_actor_idxs = []

        self.object_rigid_body_idxs = []
        self.robot_rigid_body_idxs = []
        self.feet_rigid_body_idxs = []

        self.imu_sensor_handles = []

        self.envs = []

        # Invert for easier division
        self.masses = torch.zeros(self.num_envs * self.num_actors_per_env, device=self.device)
        self.masses = self.masses[:, None]

        self.env_actor_offsets = []
        self.env_actor_rotations = []

        #todo move to load assets
        if self.cfg.env.base_texture:
            self.base_asset = self.gym.create_box(self.sim, self.cfg.env.base_dims[0], self.cfg.env.base_dims[1], self.cfg.env.base_dims[2])
            self.texture = self.gym.create_texture_from_file(self.sim, '../resources/textures/tiled_grass.jpg')
            y="""            f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/"""
            self.terrain_actors = [None] * self.num_envs
            self.goal_actors = [None] * self.num_envs * 2

            self.goal_asset = self.gym.create_box(self.sim, self.cfg.env.goal_dims[0], self.cfg.env.goal_dims[1], self.cfg.env.goal_dims[2])

            #todo create from config 
            asset_opt = gymapi.AssetOptions()
    #        asset_opt.fix_base_link=Tr
            self.ball_asset = self.gym.create_sphere(self.sim, 0.08, asset_opt)

            self.num_bodies_per_env += self.gym.get_asset_rigid_body_count(self.base_asset)
            self.num_bodies_per_env += 2 * self.gym.get_asset_rigid_body_count(self.goal_asset)
            self.num_bodies_per_env += self.gym.get_asset_rigid_body_count(self.ball_asset)

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            env_pos = self.env_origins[i].clone()
#            pos[0:1] += torch_rand_float(-self.cfg.terrain.x_init_range, self.cfg.terrain.x_init_range, (1, 1),device=self.device).squeeze(1)
#            pos[1:2] += torch_rand_float(-self.cfg.terrain.y_init_range, self.cfg.terrain.y_init_range, (1, 1), device=self.device).squeeze(1)
#            env_pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            k = 0
            #todo create aggregates for performance boost
            for asset_name, asset_pos in positions.items():
                asset = self.assets['nao']
                for j, pos in enumerate(asset_pos):
                    start_pose.p = gymapi.Vec3(*[x + y + z for x, y, z in zip(env_pos, pos, self.base_init_state[:3])])
                    if asset_name in rotations:
                        start_pose.r = gymapi.Quat(*rotations[asset_name])

                    self.env_actor_offsets.append(pos)
                    self.env_actor_rotations.append(rotations.get(asset_name, (0.,0.,0.,1.)))
                    
                    # todo, update randomization to be per actor instead of per environment?
                    rigid_shape_props = self._process_rigid_shape_props(self.asset_rigid_shape_props[asset_name], i)
                    self.gym.set_asset_rigid_shape_properties(asset, rigid_shape_props)

                    actor_handle = self.gym.create_actor(env_handle, asset, start_pose, self.cfg.asset.name + '%d'%k, i, self.cfg.asset.self_collisions.get(asset_name, 0), 0)

                    dof_props = self._process_dof_props(self.asset_dof_props[asset_name], i)
                    self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)

                    body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
                    body_props = self._process_rigid_body_props(body_props, i * self.num_actors_per_env + k)

                    self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True) 

                    if self.asset_num_dof[asset_name] > 0:
                        if i == 0:
                            self.robot_actor_handles.append(actor_handle)
                        self.robot_actor_idxs.append(self.gym.get_actor_index(env_handle, actor_handle, gymapi.DOMAIN_SIM))
                        for bi in self.asset_body_names[asset_name]:
                            self.robot_rigid_body_idxs.append(self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, bi))
                    else:
                        self.object_actor_handles.append(actor_handle)
                        self.object_actor_idxs.append(self.gym.get_actor_index(env_handle, actor_handle, gymapi.DOMAIN_SIM))
                    k += 1

            if self.cfg.env.base_texture:
                self._add_texture_terrain(env_handle, i)
                self._add_goal_boxes(env_handle, i)
                env_pos[:2] += torch_rand_float(-0.5, 0.5, (1,2), device=self.device).squeeze(0)
                start_pose.p = gymapi.Vec3(*env_pos)

                ball_handle = self.gym.create_actor(env_handle, self.ball_asset, start_pose, 'ball', i, 0, 0)
                self.ball_actor_handles.append(ball_handle)
                self.ball_actor_idxs.append(self.gym.get_actor_index(env_handle, ball_handle, gymapi.DOMAIN_SIM))
            self.envs.append(env_handle) 


        self.robot_actor_idxs = torch.tensor(self.robot_actor_idxs, device=self.device)
        self.object_actor_idxs = torch.tensor(self.object_actor_idxs, device=self.device)
        self.ball_actor_idxs = torch.tensor(self.ball_actor_idxs, device=self.device)
        print(self.ball_actor_idxs, self.robot_actor_idxs)
        self.object_rigid_body_idxs = torch.tensor(self.object_rigid_body_idxs, device=self.device)
        self.env_actor_offsets = torch.tensor(self.env_actor_offsets, device=self.device)
        self.env_actor_rotations = torch.tensor(self.env_actor_rotations, device=self.device)

        self._setup_penalties_and_terminations()

    def _add_texture_terrain(self, env, env_idx):
        pose = gymapi.Transform()
        pose.r = gymapi.Quat(0, 0, 0, 1)
        pose.p = gymapi.Vec3(*self.env_origins[env_idx])


        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        self.terrain_actors[env_idx] = self.gym.create_actor(env, self.base_asset, pose, None, self.num_envs, 0)
#        self.env.gym.set_actor_scale(env, self.env.terrain_actors[i, j], self.env.terrain.cfg.env_width)
        # self.gym.set_rigid_body_color(env, base_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
        #                          gymapi.Vec3(1, 1, 1))
        # print(len(textures), segmentation_id, segmentation_id // (255 // len(texture_files) + len(texture_files)))
        self.gym.set_rigid_body_texture(env, self.terrain_actors[env_idx], 0, gymapi.MeshType.MESH_VISUAL, self.texture)

    def _add_goal_boxes(self, env, env_idx):
        pose = gymapi.Transform()
        pose.r = gymapi.Quat(0, 0, 0, 1)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        goal_idx = env_idx * 2

        pose.p = gymapi.Vec3(self.env_origins[env_idx][0] - ((self.cfg.env.base_dims[0] + self.cfg.env.goal_dims[0]) / 2), self.env_origins[env_idx][1], self.env_origins[env_idx][2])

        self.goal_actors[goal_idx] = self.gym.create_actor(env, self.goal_asset, pose, None, self.num_envs, 0)
        self.gym.set_rigid_body_color(env, self.goal_actors[goal_idx], 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 1, 1))

        pose.p = gymapi.Vec3(self.env_origins[env_idx][0] + ((self.cfg.env.base_dims[0] + self.cfg.env.goal_dims[0]) / 2), self.env_origins[env_idx][1], self.env_origins[env_idx][2])

        self.goal_actors[goal_idx + 1] = self.gym.create_actor(env, self.goal_asset, pose, None, self.num_envs, 0)
        self.gym.set_rigid_body_color(env, self.goal_actors[goal_idx + 1], 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 1, 1))


    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(-num_rows/2, num_rows/2), torch.arange(-num_cols/2, num_cols/2))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum((1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1)).view(1,-1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices.view(-1), 2].view(self.num_envs * self.num_actors_per_env, self.feet_indices.shape[1])
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_no_fly(self):
        contacts = (self.contact_forces[:, self.feet_indices, 2]).view(self.num_envs * self.num_actors_per_env, self.feet_indices.shape[1]) > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact

    def _reward_tracking_lin_vel_ball(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.ball_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma * 2)

    def _reward_ball_distance(self):
        # Tracking of linear velocity commands (xy axes)
        left_foot_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_actor_handles[0], self.cfg.asset.feet_names[self.asset_name][0])
        left_foot_pos = self.body_states.view(self.num_envs, -1, 13)[:,self.feet_indices[0][0],0:3].view(self.num_envs,3)#-self.base_po)
        delta = self.ball_pos - left_foot_pos
#        delta_oriented = quat_rotate_inverse(self.base_quat,delta)

#        print(self.ball_pos.shape, left_foot_pos.shape)
        ball_dist_error = torch.pow(torch.norm(delta, dim=-1), 2)
        return torch.exp(-ball_dist_error/self.cfg.rewards.tracking_sigma)

#todo move helpers    
def parse_cfg_asset_options(cfg):
    asset_options = gymapi.AssetOptions()

    asset_options.default_dof_drive_mode = cfg.asset.default_dof_drive_mode
    asset_options.collapse_fixed_joints = cfg.asset.collapse_fixed_joints
    asset_options.replace_cylinder_with_capsule = cfg.asset.replace_cylinder_with_capsule
    asset_options.flip_visual_attachments = cfg.asset.flip_visual_attachments
    asset_options.fix_base_link = cfg.asset.fix_base_link
    asset_options.density = cfg.asset.density
    asset_options.angular_damping = cfg.asset.angular_damping
    asset_options.linear_damping = cfg.asset.linear_damping
    asset_options.max_angular_velocity = cfg.asset.max_angular_velocity
    asset_options.max_linear_velocity = cfg.asset.max_linear_velocity
    asset_options.armature = cfg.asset.armature
    asset_options.thickness = cfg.asset.thickness
    asset_options.disable_gravity = cfg.asset.disable_gravity

    return asset_options

def create_ball(gym, sim, radius = 0.4):
    ball_pose = gymapi.Transform()
    ball_pose.p = gymapi.Vec3(0, 0, 0)
    ball_pose.r = gymapi.Quat(0, 0, 0, 1)

    asset_options = gymapi.AssetOptions()
    ball_asset = gym.create_sphere(sim, radius, asset_options)
    return ball_asset
