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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class NaoMultiCfg( LeggedRobotCfg ): 
    class env( LeggedRobotCfg.env ):
        num_envs = 256
        num_actions = 22
        # TODO This is hardcoded for now but should be inferred
        num_observations = 78
        episode_length_s = 20
        env_spacing = 15
        base_texture = True
        base_dims = [10, 5, 0.01]
        goal_dims = [base_dims[1] / 4, base_dims[0] / 4, 0.01]
    class init_state( LeggedRobotCfg.init_state ):
#        pos = [0.0, 0.0, 0.30] # x,y,z [m]
        pos = [0.0, 0.0, 0.35] # x,y,z [m]

        use_halfway = False
        default_joint_angles = { # = target angles [rad] when action = 0.0
                                'HeadYaw': 0.0,
                                'HeadPitch': 0.0,

                                'LShoulderPitch': 1.396,
                                'LShoulderRoll': 0.1,
                                'LElbowYaw': -1.396,
                                'LElbowRoll': -0.1,

                                'RShoulderPitch': 1.396,
                                'RShoulderRoll': -0.1,
                                'RElbowYaw': 1.396,
                                'RElbowRoll': 0.1,

                                'RHipYawPitch': 0.0,
                                'RHipRoll': 0.0,
                                'RHipPitch': -0.436,
                                'RKneePitch': 0.698,
                                'RAnklePitch': 0.349,
                                'RAnkleRoll': 0.0,

                                'LHipYawPitch': 0.0,
                                'LHipRoll': 0.0,
                                'LHipPitch': -0.436,
                                'LKneePitch': 0.698,
                                'LAnklePitch': 0.349,
                                'LAnkleRoll': 0.0,
                                }
        default_joint_angles = { # = target angles [rad] when action = 0.0
                                'HeadYaw': 0.0,
                                'HeadPitch': 0.0,

                                'LShoulderPitch': 0,
                                'LShoulderRoll': 0,
                                'LElbowYaw': 0,
                                'LElbowRoll': 0,

                                'RShoulderPitch': 0,
                                'RShoulderRoll': 0,
                                'RElbowYaw': 0,
                                'RElbowRoll': 0,

                                'RHipYawPitch': 0.0,
                                'RHipRoll': 0.0,
                                'RHipPitch': 0.,
                                'RKneePitch': 0.,
                                'RAnklePitch': 0.0,
                                'RAnkleRoll': 0.0,

                                'LHipYawPitch': 0.0,
                                'LHipRoll': 0.0,
                                'LHipPitch': 0,
                                'LKneePitch': 0,
                                'LAnklePitch': 0,
                                'LAnkleRoll': 0.0,
                                }
    class terrain( LeggedRobotCfg.terrain):
#        mesh_type='trimesh'
        mesh_type = 'plane'
        measure_heights= False

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {   """        'HeadYaw': 0.75,
                                'HeadPitch': 2.6,
                                'HipYawPitch': 3.6,
                                'HipRoll':5.8,
                                'HipPitch':3.0,
                                'KneePitch':3.2,#2.8,
                                'AnklePitch':2.9,
                                'AnkleRoll':5.9,
                                'ShoulderPitch':0.65,
                                'ShoulderRoll':2.2,
                                'ElbowYaw':0.75,
                                'ElbowRoll':2.1,
                                'Finger': 0.,
                                'Thumb': 0.,""":None,
#                     'KneePitch': 25,
                     'joint': 10.}  # [N*m/rad]

        #joint: 10
        damping = {'joint': 0.1, "Finger": 0., "Thumb": 0.}     # [N*m*s/rad]

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.3
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class noise ( LeggedRobotCfg.noise ):
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales (LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.03
            ang_vel = 0.05
            gravity = 0.05
            height_measurements = 0.02

    class asset( LeggedRobotCfg.asset ):
#        positions = {'nao': [(0,0,0), (1,1,0), (0.5, 0.5, 0), (1.5, 1.5, 0)]}
        positions = {'nao': [(-1,0,0), (-4,0,0), (-3.5,2,0), (-3,-2,0)], 'nao2': [(1,0,0), (4,0,0), (3,2,0), (3,-2,0)]}
#        positions = {'nao': [(-.3,0,0), (-4,0,0), (-3.5,2,0), (-3,-2,0), (0.3,0,0), (4,0,0), (3,2,0), (3,-2,0)]}
        rotations = {}
        rotations = {'nao2': (0, 0, 1, 0)}

        asset_paths = {'nao': 'robots/nao/urdf/nao.urdf', 'nao2': 'robots/nao/urdf/nao.urdf'}
        feet_names = {'nao': 'ankle', 'nao2': 'ankle'} 
        penalize_contacts_on = {'nao': ["Knee", "Elbow"], 'nao2': ["Knee", "Elbow"]}
        terminate_after_contacts_on = {'nao': ['Hip', 'Thigh', 'Shoulder', 'Pelvis', 'Head', 'Finger', 'Elbow', 'Knee', 'Thumb', 'ForeArm', 'Tibia', 'Bicep', 'Neck', 'gripper', 'Bumper', 'Hand', 'Chest']}

        self_collisions = {'nao': 0, 'nao2': 0}
        terminate_heights_on = {'nao': [('head', 0.27)]}

        num_actors_per_env = sum([len(pos) for pos in positions.values()])

        #todo fix pathing for assets
        file = '{LEGGED_GYM_RESOURCES_DIR}/resources/robots/nao/urdf/nao.urdf'

    class commands (LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 4.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges (LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.1, 0.1]   # min max [m/s]
            ang_vel_yaw = [-0.1, 0.1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    
    class sim (LeggedRobotCfg.sim):
        dt = 0.005

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        max_contact_force = 1000.
        only_positive_rewards = False
        tracking_sigma = 0.25#0.25 # tracking reward = exp(-error^2/sigma)
        class scales( LeggedRobotCfg.rewards.scales ):
#            termination = -200.
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.2
            torques = -5.e-6
            dof_acc = 0.
            lin_vel_z = 0.
            dof_acc = 0.#2.e-7
            lin_vel_z = 0.#-0.5
            feet_air_time = 5.
            dof_pos_limits = 0.#-1.
            no_fly = 0.25
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -0.
            action_rate = 0.

        x= """
        soft_dof_pos_limit = 0.0
        soft_dof_vel_limit = 0.0
        soft_torque_limit = 0.0
        max_contact_force = 1000.
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = 0.
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.7
            torques = 0.
            dof_acc = 0.
            lin_vel_z = 0.
            dof_acc = 0.
            lin_vel_z = 0.
            feet_air_time = 0.
            dof_pos_limits = 0.
            no_fly = 0.
            dof_vel = 0.0
            ang_vel_xy = 0.0
            feet_contact_forces = 0.
            action_rate = 0.
            """

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 0.1


class NaoMultiCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 5e-4
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'nao'
        max_iterations = 1500


