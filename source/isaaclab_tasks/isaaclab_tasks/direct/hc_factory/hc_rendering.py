"""Apply HRTPA RTX render settings on top of the camera rendering kit."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.sim import RenderCfg


def _suppress_simulation_manager_plugin_warnings() -> None:
    """Suppress benign camera/render interpolation warnings (Isaac Sim #403)."""
    import carb

    carb.logging.acquire_logging().set_level_threshold_for_source(
        "isaacsim.core.simulation_manager.plugin",
        carb.logging.LogSettingBehavior.OVERRIDE,
        carb.logging.LEVEL_ERROR,
    )


def apply_hc_render_settings(render_cfg: RenderCfg) -> None:
    """Enable colored RTX output for training / video capture."""
    import carb

    settings = carb.settings.get_settings()
    if not settings.get("/isaaclab/cameras_enabled"):
        return

    _suppress_simulation_manager_plugin_warnings()

    # Kit defaults disable sampled lighting (flat grayscale). Re-enable for scene DomeLight.
    settings.set_bool("/rtx/directLighting/sampledLighting/enabled", True)
    settings.set_bool("/rtx/directLighting/enabled", True)
    settings.set_int("/rtx/directLighting/sampledLighting/samplesPerPixel", render_cfg.samples_per_pixel)
    settings.set_bool("/rtx/shadows/enabled", render_cfg.enable_shadows)
    settings.set_int("/rtx/domeLight/upperLowerStrategy", 0)
    settings.set_bool("/rtx/ambientOcclusion/enabled", render_cfg.enable_ambient_occlusion)
    settings.set_int("/rtx/post/dlss/execMode", render_cfg.dlss_mode)

    # DL denoiser / DLAA break headless replicator rgb capture (all-black frames).
    settings.set_int("/rtx/directLighting/sampledLighting/denoisingTechnique", 0)
    settings.set_bool("/rtx-transient/dldenoiser/enabled", False)

    try:
        import omni.replicator.core as rep

        rep.settings.set_render_rtx_realtime(antialiasing=render_cfg.antialiasing_mode)
    except Exception:
        pass


def enable_color_rtx_rendering() -> None:
    """Backward-compatible entry point."""
    from isaaclab.sim import RenderCfg

    apply_hc_render_settings(RenderCfg(samples_per_pixel=2, enable_ambient_occlusion=True))
