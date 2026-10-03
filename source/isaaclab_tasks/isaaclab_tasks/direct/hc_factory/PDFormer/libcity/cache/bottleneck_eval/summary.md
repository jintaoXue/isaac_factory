# Bottleneck forecast eval summary

- horizon K = 20 windows (60 s), test windows = 5354
- devices = 15: num00_rotaryPipeAutomaticWeldingMachine_ws0, num00_rotaryPipeAutomaticWeldingMachine_ws1, num01_weldingRobot_ws0, num02_rollerbedCNCPipeIntersectionCuttingMachine_ws0, num04_groovingMachineLarge_ws0, num08_workbench_ws0, num08_workbench_ws1, gantry_0, gantry_1, gantry_2, gantry_3, robot_0, robot_1, robot_2, robot_3
- key device = `num02_rollerbedCNCPipeIntersectionCuttingMachine_ws0` (picked on `dense_i1_12_3_start5_min8_cause4_opt_main_seed42`), plot episode = `n10_human1.0__episode_03`, all-device heatmap episode = `n10_human1.0__episode_03`

Durations / starts in minutes. `dur MAE/RMSE` = over every (window, device) where a bottleneck is true or predicted (missed → pred 0, false alarm → true 0); `tp` = correctly detected bottlenecks only. State = bottleneck yes/no per future window.

| run | thr | P | R | will15 F1 | ref F1 | state acc 1step | state F1 1step | state F1 K | dur MAE | dur RMSE | dur MAE tp | dur RMSE tp | start MAE tp | start RMSE tp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ablation_nograph_start10 | 0.94 | 0.821 | 0.539 | 0.651 | 0.651 | 0.996 | 0.844 | 0.394 | 6.46 | 7.91 | 3.09 | 4.45 | 0.05 | 0.55 |
| ablation_nograph_start15 | 0.94 | 0.851 | 0.482 | 0.616 | 0.616 | 0.997 | 0.865 | 0.389 | 6.57 | 7.94 | 3.04 | 4.38 | 0.07 | 0.80 |
| ablation_nograph_start5 | 0.94 | 0.832 | 0.678 | 0.747 | 0.747 | 0.997 | 0.873 | 0.400 | 5.48 | 7.12 | 2.98 | 4.26 | 0.02 | 0.23 |
| ablation_nogroup_start10 | 0.94 | 0.605 | 0.745 | 0.668 | 0.668 | 0.995 | 0.826 | 0.464 | 6.05 | 7.48 | 2.43 | 3.40 | 0.46 | 1.17 |
| ablation_nogroup_start15 | 0.94 | 0.567 | 0.690 | 0.623 | 0.623 | 0.995 | 0.819 | 0.442 | 6.34 | 7.64 | 2.49 | 3.46 | 0.52 | 1.32 |
| ablation_nogroup_start5 | 0.94 | 0.766 | 0.756 | 0.761 | 0.761 | 0.996 | 0.835 | 0.440 | 5.11 | 6.68 | 2.59 | 3.53 | 0.17 | 0.67 |
| 12_3_start10 | 0.94 | 0.806 | 0.761 | 0.783 | 0.783 | 0.997 | 0.892 | 0.589 | 4.38 | 6.11 | 1.79 | 2.50 | 0.23 | 0.74 |
| 12_3_start15 | 0.94 | 0.806 | 0.742 | 0.772 | 0.772 | 0.998 | 0.900 | 0.610 | 4.43 | 6.15 | 1.72 | 2.49 | 0.23 | 0.81 |
| 12_3_start5 | 0.94 | 0.794 | 0.817 | 0.805 | 0.805 | 0.997 | 0.883 | 0.525 | 4.17 | 5.92 | 1.86 | 2.56 | 0.13 | 0.43 |
| a1_prefix8 | 0.98 | 0.852 | 0.856 | 0.854 | - | 0.997 | 0.868 | 0.547 | 3.39 | 4.85 | 1.96 | 2.60 | 0.09 | 0.38 |
| entity_machineonly_start10 | 0.94 | 0.820 | 0.424 | 0.559 | 0.559 | 0.994 | 0.708 | 0.311 | 7.40 | 8.62 | 3.60 | 5.00 | 0.00 | 0.04 |
| entity_machineonly_start15 | 0.94 | 0.802 | 0.400 | 0.534 | 0.534 | 0.994 | 0.706 | 0.317 | 7.46 | 8.58 | 3.39 | 4.60 | 0.02 | 0.17 |
| entity_machineonly_start5 | 0.94 | 0.798 | 0.549 | 0.650 | 0.650 | 0.994 | 0.709 | 0.312 | 6.51 | 7.96 | 3.57 | 5.06 | 0.01 | 0.14 |
| entity_nocross_start10 | 0.94 | 0.596 | 0.593 | 0.595 | 0.595 | 0.993 | 0.759 | 0.388 | 6.60 | 7.96 | 2.51 | 3.62 | 0.34 | 1.22 |
| entity_nocross_start15 | 0.94 | 0.518 | 0.616 | 0.563 | 0.563 | 0.993 | 0.760 | 0.379 | 6.77 | 8.03 | 2.37 | 3.46 | 0.73 | 1.90 |
| entity_nocross_start5 | 0.94 | 0.750 | 0.698 | 0.723 | 0.723 | 0.995 | 0.812 | 0.407 | 5.28 | 6.87 | 2.59 | 3.80 | 0.11 | 0.57 |
| entity_noinfo_start10 | 0.94 | 0.759 | 0.425 | 0.545 | 0.545 | 0.993 | 0.684 | 0.306 | 7.57 | 8.75 | 3.72 | 5.15 | 0.02 | 0.36 |
| entity_noinfo_start15 | 0.94 | 0.466 | 0.518 | 0.490 | 0.490 | 0.989 | 0.596 | 0.309 | 7.66 | 8.72 | 2.94 | 4.25 | 0.77 | 2.16 |
| entity_noinfo_start5 | 0.94 | 0.770 | 0.553 | 0.644 | 0.644 | 0.993 | 0.699 | 0.311 | 6.66 | 8.10 | 3.55 | 5.07 | 0.02 | 0.18 |

## Per-device (12_3_start5)

| device | type | n_true | n_pred | P | R | F1 | dur MAE | dur RMSE | dur MAE tp | dur RMSE tp | state acc 1step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| num00_rotaryPipeAutomaticWeldingMachine_ws0 | machine | 115 | 100 | 0.700 | 0.609 | 0.651 | 5.31 | 7.11 | 1.37 | 1.55 | 0.994 |
| num00_rotaryPipeAutomaticWeldingMachine_ws1 | machine | 17 | 19 | 0.895 | 1.000 | 0.944 | 1.54 | 2.76 | 1.03 | 1.25 | 0.999 |
| num01_weldingRobot_ws0 | machine | 46 | 25 | 0.760 | 0.413 | 0.535 | 6.04 | 7.56 | 1.19 | 1.24 | 0.997 |
| num02_rollerbedCNCPipeIntersectionCuttingMachine_ws0 | machine | 301 | 343 | 0.767 | 0.874 | 0.817 | 4.67 | 6.45 | 2.28 | 2.93 | 0.989 |
| num04_groovingMachineLarge_ws0 | machine | 20 | 27 | 0.556 | 0.750 | 0.638 | 5.32 | 6.87 | 1.43 | 1.67 | 0.999 |
| num08_workbench_ws0 | machine | 186 | 162 | 0.914 | 0.796 | 0.851 | 3.47 | 5.02 | 1.91 | 2.99 | 0.995 |
| num08_workbench_ws1 | machine | 94 | 101 | 0.931 | 1.000 | 0.964 | 2.65 | 3.65 | 2.40 | 3.18 | 0.999 |
| gantry_0 | gantry | 18 | 26 | 0.577 | 0.833 | 0.682 | 3.35 | 4.85 | 1.40 | 1.67 | 0.998 |
| gantry_1 | gantry | 31 | 34 | 0.765 | 0.839 | 0.800 | 4.55 | 6.45 | 1.57 | 1.81 | 1.000 |
| gantry_2 | gantry | 108 | 132 | 0.742 | 0.907 | 0.817 | 3.59 | 5.56 | 1.20 | 1.78 | 0.996 |
| gantry_3 | gantry | 121 | 127 | 0.811 | 0.851 | 0.831 | 3.56 | 5.04 | 1.66 | 2.02 | 0.990 |
| robot_0 | transport_robot | 15 | 10 | 0.900 | 0.600 | 0.720 | 4.01 | 6.14 | 0.30 | 0.39 | 1.000 |
| robot_1 | transport_robot | 30 | 28 | 0.821 | 0.767 | 0.793 | 5.03 | 6.80 | 2.48 | 3.27 | 0.999 |
| robot_2 | transport_robot | 0 | 0 | 0.000 | 0.000 | 0.000 | nan | nan | nan | nan | 1.000 |
| robot_3 | transport_robot | 0 | 0 | 0.000 | 0.000 | 0.000 | nan | nan | nan | nan | 1.000 |
