# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Human-robot task allocation for production environment.
"""

import gymnasium as gym

from . import algo_cfg

##
# Register Gym environments.
##

gym.register(
    order_enforce=False,
    id="HRTPaHC-v1",
    entry_point=f"{__name__}.hc_vector_env:HcVectorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfgs.hc_env_cfg:HcEnvCfg",
        "rule_based": f"{algo_cfg.__name__}:rule_based.yaml",
        "rl_filter": f"{algo_cfg.__name__}:rl_filter.yaml",
    },
)

