"""MuJoCo simulation robot that implements the i2rt Robot protocol.

Wraps a MuJoCo model and provides forward-kinematics visualization
driven by joint position commands. Optionally launches a passive viewer.
"""

import time
from typing import Dict, List, Optional

import mujoco
import mujoco.viewer
import numpy as np
from i2rt.robots.robot import Robot


class MujocoSimRobot(Robot):
    """A simulated robot backed by a MuJoCo model.

    Accepts joint-position commands (in radians), updates the model state
    via ``mj_kinematics``, and optionally renders in a passive MuJoCo
    viewer window.

    Args:
        xml_path: Path to the MuJoCo XML model file.
        render: Whether to launch a passive viewer window.
        gripper_index: If set, the last DOF in ``command_joint_pos``
            is treated as a virtual gripper value (not part of the
            MuJoCo model's qpos).
        camera_obs: Enable offscreen camera rendering in observations.
        camera_width: Width of offscreen-rendered camera images.
        camera_height: Height of offscreen-rendered camera images.
        camera_pos: Fixed camera position [x, y, z] in world frame.
        camera_lookat: Point the camera looks at [x, y, z].
    """

    def __init__(
        self,
        xml_path: str,
        render: bool = True,
        gripper_index: Optional[int] = None,
        camera_obs: bool = False,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_pos: Optional[List[float]] = None,
        camera_lookat: Optional[List[float]] = None,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._gripper_index = gripper_index
        self._gripper_pos = np.array([0.0])

        # Total DOFs exposed to the control system
        self._nq = self.model.nq
        self._num_dofs = self._nq + (1 if gripper_index is not None else 0)

        # Offscreen camera rendering
        self._camera_obs = camera_obs
        self._offscreen_renderer: Optional[mujoco.Renderer] = None
        self._cam = mujoco.MjvCamera()
        if camera_obs:
            self._offscreen_renderer = mujoco.Renderer(
                self.model, height=camera_height, width=camera_width,
            )
            cam_pos = camera_pos or [0.6, -0.3, 0.6]
            lookat = camera_lookat or [0.0, 0.0, 0.15]
            self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._cam.lookat[:] = lookat
            direction = np.array(cam_pos) - np.array(lookat)
            self._cam.distance = float(np.linalg.norm(direction))
            self._cam.azimuth = float(np.degrees(np.arctan2(-direction[1], direction[0])))
            self._cam.elevation = float(
                np.degrees(np.arcsin(direction[2] / (self._cam.distance + 1e-8)))
            )

        # Initialize scene state for rendering
        mujoco.mj_forward(self.model, self.data)

        # Optionally launch viewer
        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(
                model=self.model,
                data=self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)

    # ------------------------------------------------------------------ #
    # Robot protocol
    # ------------------------------------------------------------------ #

    def num_dofs(self) -> int:
        return self._num_dofs

    def get_joint_pos(self) -> np.ndarray:
        qpos = self.data.qpos[: self._nq].copy()
        if self._gripper_index is not None:
            return np.concatenate([qpos, self._gripper_pos])
        return qpos

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Set the sim model's qpos and update kinematics + viewer.

        Args:
            joint_pos: Joint positions in radians.  If ``gripper_index``
                was provided, the last element is the gripper value and
                the preceding elements map to MuJoCo joints.
        """
        self.data.qpos[: self._nq] = joint_pos[: self._nq]
        if self._gripper_index is not None and len(joint_pos) > self._nq:
            self._gripper_pos[0] = joint_pos[self._nq]

        mujoco.mj_kinematics(self.model, self.data)

        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def get_observations(self) -> Dict[str, np.ndarray]:
        obs: Dict[str, np.ndarray] = {
            "joint_pos": self.data.qpos[: self._nq].copy(),
            "joint_vel": self.data.qvel[: self._nq].copy(),
        }
        if self._gripper_index is not None:
            obs["gripper_pos"] = self._gripper_pos.copy()
        if self._camera_obs and self._offscreen_renderer is not None:
            self._offscreen_renderer.update_scene(self.data, self._cam)
            frame = self._offscreen_renderer.render()
            obs["sim_camera"] = {
                "images": {"rgb": frame.copy()},
                "timestamp": time.time(),
            }
        return obs

    # ------------------------------------------------------------------ #
    # Viewer helpers
    # ------------------------------------------------------------------ #

    def is_viewer_running(self) -> bool:
        """Return True if the viewer window is still open."""
        if self.viewer is None:
            return True  # headless mode — never "closes"
        return self.viewer.is_running()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
