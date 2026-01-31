# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Human-robot task allocation for production environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    order_enforce=False,
    id="HRTPaHC-v1",
    entry_point=f"{__name__}.hc_env:HcEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hc_env_cfg:HcEnvCfg",
        "rl_filter": f"{agents.__name__}:rl_filter.yaml",
        "ppo_dis": f"{agents.__name__}:ppo_dis.yaml",
        "ppolag_filter_dis": f"{agents.__name__}:ppolag_filter_dis.yaml",
        "dqn": f"{agents.__name__}:dqn.yaml",
        "cpo_filter": f"{agents.__name__}:cpo_filter.yaml",
        "rl_filter_mlp": f"{agents.__name__}:rl_filter_mlp.yaml",
        "rl_filter_selfattn": f"{agents.__name__}:rl_filter_selfattn.yaml",
        "rl_filter_no_noisy": f"{agents.__name__}:rl_filter_no_noisy.yaml",
        "rl_filter_no_dueling": f"{agents.__name__}:rl_filter_no_dueling.yaml",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

