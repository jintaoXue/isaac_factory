from isaaclab.envs.ui.base_env_window import BaseEnvWindow


class HcEnvWindow(BaseEnvWindow):
    """Direct-workflow env window without ManagerBased action/observation panels."""

    # def _visualize_manager(self, title: str, class_name: str) -> None:
    #     if class_name in {"action_manager", "observation_manager"}:
    #         return
    #     super()._visualize_manager(title, class_name)
