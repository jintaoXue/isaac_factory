"""Gantry zone ownership: resource name -> gantry index (0..3).

Used to decide same-zone vs cross-zone after a processing goal is known.
Lookup is by resource name (not pure X bands), matching the agreed layout.
"""

# Final ownership (g0/g1/g2/g3). num08 stations split across g0/g1.
RESOURCE_TO_GANTRY_ZONE: dict[str, int] = {
    # g0
    "num00_rotaryPipeAutomaticWeldingMachine": 0,
    "num08_workbench_station_00": 0,
    "BlackStorage_00": 0,
    "BlackStorage_01": 0,
    "GroundStorage_00": 0,
    # g1
    "num08_workbench_station_01": 1,
    "num01_weldingRobot": 1,
    "num02_rollerbedCNCPipeIntersectionCuttingMachine": 1,
    "num03_laserCuttingMachine": 1,
    "BlackStorage_02": 1,
    "GroundStorage_01": 1,
    # g2
    "BlackStorage_03": 2,
    "BlackStorage_04": 2,
    "BlackStorage_05": 2,
    "YellowStorage_00": 2,
    "YellowStorage_01": 2,
    "YellowStorage_02": 2,
    "YellowStorage_03": 2,
    "YellowStorage_04": 2,
    "num04_groovingMachineLarge": 2,
    "num05_groovingMachineSmall": 2,
    # g3
    "YellowStorage_05": 3,
    "YellowStorage_06": 3,
    "YellowStorage_07": 3,
    "YellowStorage_08": 3,
    "YellowStorage_09": 3,
    "YellowStorage_10": 3,
    "num06_highPressureFoamingMachine": 3,
}


def get_gantry_zone(resource_name: str | None, workstation_key: str | None = None) -> int | None:
    """Return gantry zone id for a machine/storage name.

    For ``num08_workbench``, pass the station key (``num08_workbench_station_0x``).
    """
    if resource_name is None:
        return None
    if resource_name == "num08_workbench":
        if workstation_key is not None and workstation_key in RESOURCE_TO_GANTRY_ZONE:
            return RESOURCE_TO_GANTRY_ZONE[workstation_key]
        return None
    return RESOURCE_TO_GANTRY_ZONE.get(resource_name)
