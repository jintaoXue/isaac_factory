# Recon blocked/starved eval summary

- window = 60 s, horizon K = 20 windows
- all devices = 15 supervised nodes: num00_rotaryPipeAutomaticWeldingMachine_ws0, num00_rotaryPipeAutomaticWeldingMachine_ws1, num01_weldingRobot_ws0, num02_rollerbedCNCPipeIntersectionCuttingMachine_ws0, num04_groovingMachineLarge_ws0, num08_workbench_ws0, num08_workbench_ws1, gantry_0, gantry_1, gantry_2, gantry_3, robot_0, robot_1, robot_2, robot_3
- key device = `num01_weldingRobot_ws0` (picked on `dense_i1_12_3_start5_min8_cause4_opt_main_seed42`)
- key episode for plots = `logistics20__episode_16`

Units: seconds per window. 1step = first future window; horizon = mean over all valid K steps; persist = last observed window copied forward (sanity floor, 1 step).

| run | Blocked MAE 1step | Blocked RMSE 1step | Starved MAE 1step | Starved RMSE 1step | Blocked MAE hor | Blocked RMSE hor | Starved MAE hor | Starved RMSE hor | Key Blocked MAE/RMSE | Key Starved MAE/RMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| ablation_nograph_start10 | 0.53 | 3.42 | 10.39 | 18.05 | 0.51 | 3.39 | 10.53 | 18.40 | 1.51/5.89 | 11.50/21.10 |
| ablation_nograph_start15 | 0.52 | 3.42 | 10.29 | 17.89 | 0.51 | 3.39 | 10.65 | 18.26 | 1.45/5.90 | 11.22/21.13 |
| ablation_nograph_start5 | 0.53 | 3.42 | 10.20 | 17.13 | 0.51 | 3.39 | 10.63 | 17.91 | 1.48/5.89 | 10.64/19.73 |
| ablation_nogroup_start10 | 0.51 | 3.42 | 7.78 | 13.93 | 0.50 | 3.39 | 10.03 | 17.53 | 1.45/5.89 | 6.56/13.38 |
| ablation_nogroup_start15 | 0.51 | 3.42 | 7.88 | 14.06 | 0.49 | 3.39 | 10.12 | 17.56 | 1.44/5.90 | 6.90/13.36 |
| ablation_nogroup_start5 | 0.50 | 3.42 | 8.50 | 14.71 | 0.49 | 3.39 | 10.09 | 17.63 | 1.45/5.90 | 7.88/14.63 |
| 12_3_start10 | 0.52 | 3.42 | 8.93 | 14.37 | 0.51 | 3.39 | 10.38 | 17.50 | 1.46/5.89 | 8.03/12.92 |
| 12_3_start15 | 0.53 | 3.42 | 8.78 | 14.70 | 0.52 | 3.39 | 10.30 | 17.59 | 1.49/5.89 | 7.64/13.71 |
| 12_3_start5 | 0.51 | 3.42 | 8.53 | 14.43 | 0.50 | 3.39 | 10.21 | 17.54 | 1.45/5.89 | 7.30/12.90 |
| a1_prefix8 | 0.51 | 3.42 | 8.17 | 13.99 | 0.50 | 3.39 | 10.15 | 17.31 | 1.46/5.89 | 6.99/11.97 |
| entity_machineonly_start10 | 1.13 | 3.50 | 12.37 | 18.43 | 1.11 | 3.46 | 12.56 | 18.69 | 1.57/5.88 | 11.99/20.85 |
| entity_machineonly_start15 | 1.18 | 3.50 | 12.62 | 18.07 | 1.13 | 3.46 | 12.83 | 18.52 | 1.61/5.86 | 12.60/19.96 |
| entity_machineonly_start5 | 1.21 | 3.50 | 12.58 | 18.26 | 1.18 | 3.47 | 12.78 | 18.57 | 1.70/5.85 | 12.47/20.60 |
| entity_nocross_start10 | 0.51 | 3.42 | 10.05 | 16.50 | 0.50 | 3.39 | 10.76 | 17.84 | 1.43/5.90 | 10.33/18.48 |
| entity_nocross_start15 | 0.50 | 3.42 | 10.42 | 16.60 | 0.49 | 3.39 | 10.95 | 17.94 | 1.44/5.90 | 10.36/18.09 |
| entity_nocross_start5 | 0.49 | 3.42 | 9.93 | 16.76 | 0.49 | 3.39 | 10.54 | 17.85 | 1.42/5.90 | 11.19/19.33 |
| entity_noinfo_start10 | 1.10 | 3.48 | 12.20 | 18.51 | 1.04 | 3.45 | 12.32 | 18.75 | 1.55/5.88 | 12.37/20.94 |
| entity_noinfo_start15 | 1.16 | 3.49 | 12.12 | 17.10 | 1.13 | 3.46 | 12.74 | 18.35 | 1.64/5.86 | 11.32/18.46 |
| entity_noinfo_start5 | 1.17 | 3.49 | 12.43 | 17.82 | 1.14 | 3.45 | 12.69 | 18.52 | 1.64/5.87 | 12.24/19.57 |

## Active windows only (true > 0) and R2, all devices, 1step

| run | Blocked MAE act | Blocked RMSE act | Blocked R2 | Starved MAE act | Starved RMSE act | Starved R2 | Machine Starved MAE/RMSE | Machine Starved R2 |
|---|---|---|---|---|---|---|---|---|
| ablation_nograph_start10 | 20.01 | 23.02 | -0.004 | 25.94 | 32.48 | 0.076 | 9.86/19.93 | 0.018 |
| ablation_nograph_start15 | 20.03 | 23.04 | -0.006 | 25.71 | 32.16 | 0.092 | 9.85/19.75 | 0.035 |
| ablation_nograph_start5 | 20.01 | 23.03 | -0.004 | 24.12 | 30.38 | 0.167 | 9.45/18.55 | 0.149 |
| ablation_nogroup_start10 | 20.03 | 23.04 | -0.005 | 19.24 | 24.46 | 0.449 | 5.75/12.48 | 0.615 |
| ablation_nogroup_start15 | 20.04 | 23.05 | -0.006 | 19.16 | 24.56 | 0.439 | 6.06/12.44 | 0.617 |
| ablation_nogroup_start5 | 20.04 | 23.05 | -0.006 | 20.31 | 25.85 | 0.386 | 6.96/14.07 | 0.510 |
| 12_3_start10 | 19.99 | 23.01 | -0.003 | 19.06 | 24.17 | 0.414 | 7.07/13.03 | 0.580 |
| 12_3_start15 | 19.97 | 22.99 | -0.001 | 20.06 | 25.28 | 0.387 | 7.03/13.81 | 0.528 |
| 12_3_start5 | 20.00 | 23.01 | -0.003 | 19.54 | 24.82 | 0.409 | 6.46/13.19 | 0.569 |
| a1_prefix8 | 19.98 | 23.00 | -0.002 | 18.69 | 23.83 | 0.445 | 6.17/12.41 | 0.619 |
| entity_machineonly_start10 | 19.91 | 22.93 | -0.051 | 27.24 | 32.53 | 0.036 | 10.49/19.31 | 0.077 |
| entity_machineonly_start15 | 19.83 | 22.86 | -0.051 | 26.17 | 31.39 | 0.073 | 10.94/18.80 | 0.126 |
| entity_machineonly_start5 | 19.80 | 22.84 | -0.053 | 26.73 | 31.96 | 0.054 | 10.90/19.07 | 0.101 |
| entity_nocross_start10 | 20.03 | 23.04 | -0.006 | 22.80 | 28.77 | 0.228 | 9.42/17.28 | 0.261 |
| entity_nocross_start15 | 20.02 | 23.03 | -0.005 | 23.30 | 28.95 | 0.218 | 9.32/17.18 | 0.270 |
| entity_nocross_start5 | 20.05 | 23.06 | -0.007 | 23.08 | 29.45 | 0.203 | 9.27/17.87 | 0.210 |
| entity_noinfo_start10 | 19.84 | 22.88 | -0.039 | 27.94 | 33.17 | 0.028 | 11.27/19.55 | 0.054 |
| entity_noinfo_start15 | 19.78 | 22.82 | -0.045 | 24.73 | 29.25 | 0.171 | 9.87/16.41 | 0.334 |
| entity_noinfo_start5 | 19.79 | 22.83 | -0.046 | 26.16 | 31.08 | 0.099 | 10.70/18.08 | 0.192 |

Persistence floor (all devices, 1step): blocked 0.76/4.49, starved 6.54/14.93 (MAE/RMSE)

## Per-device (12_3_start5, 1step)

| device | type | Blocked MAE | Blocked RMSE | Blocked R2 | Starved MAE | Starved RMSE | Starved R2 | true std B/S |
|---|---|---|---|---|---|---|---|---|
| num00_rotaryPipeAutomaticWeldingMachine_ws0 | machine | 1.53 | 6.03 | -0.042 | 10.05 | 17.02 | 0.499 | 5.9/24.1 |
| num00_rotaryPipeAutomaticWeldingMachine_ws1 | machine | 0.29 | 2.32 | -0.001 | 2.65 | 8.67 | 0.372 | 2.3/10.9 |
| num01_weldingRobot_ws0 | machine | 1.45 | 5.89 | -0.039 | 7.30 | 12.90 | 0.618 | 5.8/20.9 |
| num02_rollerbedCNCPipeIntersectionCuttingMachine_ws0 | machine | 0.05 | 0.08 | nan | 2.12 | 7.07 | 0.706 | 0.0/13.0 |
| num04_groovingMachineLarge_ws0 | machine | 1.30 | 5.62 | -0.033 | 7.49 | 14.28 | 0.574 | 5.5/21.9 |
| num08_workbench_ws0 | machine | 1.91 | 6.81 | -0.053 | 8.65 | 14.52 | 0.596 | 6.6/22.8 |
| num08_workbench_ws1 | machine | 0.94 | 4.57 | -0.017 | 6.96 | 14.92 | 0.455 | 4.5/20.2 |
| gantry_0 | gantry | 0.01 | 0.02 | nan | 12.92 | 15.97 | -0.034 | 0.0/15.7 |
| gantry_1 | gantry | 0.00 | 0.00 | nan | 12.10 | 14.39 | 0.205 | 0.0/16.1 |
| gantry_2 | gantry | 0.02 | 0.03 | nan | 11.90 | 14.53 | 0.264 | 0.0/16.9 |
| gantry_3 | gantry | 0.07 | 0.09 | nan | 5.41 | 7.18 | 0.725 | 0.0/13.7 |
| robot_0 | transport_robot | 0.01 | 0.02 | nan | 14.01 | 20.68 | -0.064 | 0.0/20.0 |
| robot_1 | transport_robot | 0.01 | 0.03 | nan | 13.47 | 19.78 | 0.008 | 0.0/19.9 |
| robot_2 | transport_robot | 0.02 | 0.03 | nan | 8.24 | 15.88 | 0.050 | 0.0/16.3 |
| robot_3 | transport_robot | 0.03 | 0.05 | nan | 4.70 | 10.47 | 0.120 | 0.0/11.2 |
