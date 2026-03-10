"""Quest 3 / 3S VR controller teleoperation agent for I2RT YAM.

Streams controller pose and button data from a Meta Quest headset via
Vuer's WebXR ``CONTROLLER_MOVE`` events, converts to robot end-effector
targets, runs IK through the existing PyRoki solver, and outputs joint
commands compatible with both simulation and real I2RT YAM hardware.

Optionally streams camera images (MuJoCo sim or real I2RT cameras) back
to the Quest headset as a VR background via Vuer ``ImageBackground``.

Requirements:
  - ``vuer`` Python package (``uv pip install vuer``)
  - HTTPS tunnel for WebXR (``ngrok http 8012`` or ``npx localtunnel --port 8012``)
  - Quest browser navigating to the HTTPS tunnel URL

Button mapping (Quest Touch controllers):
  - A button (right) / X button (left): close gripper
  - B button (right) / Y button (left): open gripper
  - Trigger: engage teleoperation (controller pose drives the arm)
"""

import asyncio
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy
from dm_env.specs import Array

from robots_realtime.agents.agent import Agent
from robots_realtime.sensors.cameras.camera_utils import obs_get_rgb
from robots_realtime.utils.portal_utils import remote

logger = logging.getLogger(__name__)

# YAM joint limits from yam.xml — used for velocity clamping and danger zones
_YAM_JOINT_LIMITS: List[Tuple[float, float]] = [
    (-2.61799, 3.13),   # joint1
    (0.0, 3.65),        # joint2
    (0.0, 3.13),        # joint3
    (-1.65, 1.65),      # joint4
    (-1.5708, 1.5708),  # joint5
    (-2.0944, 2.0944),  # joint6
]

_DANGER_ZONE_RAD = 0.15
_MAX_JOINT_VEL_RAD_PER_S = 1.5
_GRIPPER_OPEN = 0.0
_GRIPPER_CLOSED = 2.4


def _webxr_to_robot_pos(pos_webxr: np.ndarray) -> np.ndarray:
    """Convert WebXR Y-up position to robot Z-up frame.

    Mapping (user stands behind robot, facing the same direction):
      - User forward  (WebXR -Z) → Robot +X  (arm extends forward)
      - User right    (WebXR +X) → Robot -Y  (arm sweeps right)
      - User up       (WebXR +Y) → Robot +Z  (arm lifts up)
    """
    x, y, z = pos_webxr
    return np.array([-z, -x, y])


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for unit quaternion) in wxyz format."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _webxr_mat_to_robot_quat(mat16: List[float]) -> np.ndarray:
    """Extract orientation from WebXR column-major 4x4 matrix and convert
    to robot-frame wxyz quaternion.

    Uses a frame rotation that aligns the controller's pitch/yaw/roll
    axes with the EE's intuitive tilt/swivel/roll axes:
      - Controller pitch (WebXR X) → gripper tilt up/down
      - Controller yaw   (WebXR Y) → gripper swivel left/right
      - Controller roll  (WebXR Z) → gripper axial roll
    """
    from scipy.spatial.transform import Rotation

    m = np.array(mat16, dtype=np.float64).reshape(4, 4, order="F")
    rot_webxr = m[:3, :3].copy()
    # Orientation frame: maps controller axes to EE-compatible axes.
    # WebXR Z → Robot X, WebXR X → Robot Y, WebXR Y → Robot Z
    ori_frame = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    rot_robot = ori_frame @ rot_webxr @ ori_frame.T
    quat_xyzw = Rotation.from_matrix(rot_robot).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def _extract_pos_quat_from_col_major(mat16: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract position and quaternion from a column-major 4x4 matrix.

    Returns (position_xyz, quaternion_xyzw) in the original coordinate system.
    """
    m = np.array(mat16, dtype=np.float64).reshape(4, 4, order="F")
    pos = m[:3, 3].copy()
    rot = m[:3, :3].copy()
    # Rotation matrix to quaternion (xyzw)
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    quat_xyzw = np.array([x, y, z, w])
    quat_xyzw /= np.linalg.norm(quat_xyzw)
    return pos, quat_xyzw


class _CriticallyDampedFilter:
    """Second-order critically-damped low-pass filter.

    Produces smooth, overshoot-free transitions with natural acceleration
    and deceleration -- the same trajectory-smoothing behaviour that
    commercial teleoperation stacks (Sentinel / Open-TeleVision) use.
    """

    def __init__(self, omega: float, dim: int):
        self._omega = omega       # natural frequency (higher = faster response)
        self._x: Optional[np.ndarray] = None   # filtered value
        self._v: Optional[np.ndarray] = None    # velocity
        self._dim = dim

    def reset(self, value: np.ndarray) -> None:
        self._x = value.copy()
        self._v = np.zeros(self._dim)

    def step(self, target: np.ndarray, dt: float) -> np.ndarray:
        if self._x is None:
            self.reset(target)
            return self._x.copy()
        w = self._omega
        # Critically-damped spring: damping_ratio = 1
        accel = w * w * (target - self._x) - 2.0 * w * self._v
        self._v = self._v + accel * dt
        self._x = self._x + self._v * dt
        return self._x.copy()


def _deadzone(delta: np.ndarray, threshold: float) -> np.ndarray:
    """Zero out deltas smaller than *threshold* (Euclidean norm).

    Filters micro-tremors from the user's hand so the robot stays
    rock-steady when the user intends to hold still.
    """
    if np.linalg.norm(delta) < threshold:
        return np.zeros_like(delta)
    return delta


def _clamp_to_limits(
    joints: np.ndarray,
    limits: List[Tuple[float, float]],
    danger_margin: float = _DANGER_ZONE_RAD,
) -> np.ndarray:
    """Clamp joint values to stay within [lower + margin, upper - margin]."""
    clamped = joints.copy()
    for i, (lo, hi) in enumerate(limits):
        safe_lo = lo + danger_margin
        safe_hi = hi - danger_margin
        if safe_lo >= safe_hi:
            safe_lo = (lo + hi) / 2
            safe_hi = safe_lo
        clamped[i] = np.clip(clamped[i], safe_lo, safe_hi)
    return clamped


def _limit_joint_velocity(
    new_joints: np.ndarray,
    prev_joints: np.ndarray,
    dt: float,
    max_vel: float = _MAX_JOINT_VEL_RAD_PER_S,
) -> np.ndarray:
    """Clamp per-joint velocity to max_vel rad/s."""
    if dt <= 0:
        return new_joints
    delta = new_joints - prev_joints
    max_delta = max_vel * dt
    clipped = np.clip(delta, -max_delta, max_delta)
    return prev_joints + clipped


@jdc.jit
def _solve_ik_teleop(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    prev_cfg: jax.Array,
) -> jax.Array:
    """IK solver tuned for teleoperation.

    Combines three objectives:
    - **Pose**: reach the desired position (weight 50) and orientation (weight 5).
    - **Regularization**: light bias toward the previous configuration to
      discourage unnecessary branch-jumping (weight 0.3 -- low enough to
      allow full elbow bending).
    - **Joint limits**: hard constraint to respect actuator range.
    """
    joint_var = robot.joint_var_cls(0)
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_position
            ),
            target_link_index,
            pos_weight=50.0,
            ori_weight=5.0,
        ),
        pk.costs.rest_cost(joint_var, rest_pose=prev_cfg, weight=0.3),
        pk.costs.limit_constraint(robot, joint_var),
    ]
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            initial_vals=jaxls.VarValues.make([joint_var.with_value(prev_cfg)]),
        )
    )
    return sol[joint_var]


class QuestVRAgent(Agent):
    """Teleoperation agent driven by Quest 3 / 3S controllers via Vuer WebXR.

    Uses **delta-based control**: when the trigger is first squeezed, the
    controller position and the robot end-effector position are recorded as
    anchors.  Subsequent hand movements are applied as scaled offsets from
    the EE anchor, giving intuitive 1:1 spatial control regardless of where
    the user is standing in the room.

    Args:
        bimanual: Drive both arms (left controller -> left arm, right -> right).
        vuer_port: Port for the Vuer WebXR server.
        position_scale: Multiplier applied to the controller delta.
        smoothing_alpha: EMA factor for target smoothing (0 = frozen, 1 = instant).
        max_joint_vel: Maximum joint velocity in rad/s.
        danger_zone_margin: Radians to stay away from hard joint limits.
        track_orientation: If True, map controller rotation to EE orientation.
            If False (default), use a fixed downward orientation for stability.
        default_orientation_wxyz: Fixed EE orientation used when *track_orientation*
            is False (wxyz quaternion).
    """

    def __init__(
        self,
        bimanual: bool = False,
        bimanual_combined_key: Optional[str] = None,
        vuer_port: int = 8012,
        position_scale: float = 1.0,
        smoothing_omega: float = 12.0,
        smoothing_alpha: float = 0.3,
        max_joint_vel: float = _MAX_JOINT_VEL_RAD_PER_S,
        danger_zone_margin: float = _DANGER_ZONE_RAD,
        deadzone_m: float = 0.002,
        track_orientation: bool = False,
        default_orientation_wxyz: Optional[List[float]] = None,
        stream_camera: bool = False,
        camera_key: Optional[str] = None,
        stream_fps: float = 30.0,
        stream_quality: int = 80,
        # kept for backward-compat but ignored by delta control
        position_offset: Optional[List[float]] = None,
    ) -> None:
        self.bimanual = bimanual
        self.bimanual_combined_key = bimanual_combined_key
        self.vuer_port = vuer_port
        self.position_scale = position_scale
        self._smoothing_omega = smoothing_omega
        self.smoothing_alpha = smoothing_alpha
        self.max_joint_vel = max_joint_vel
        self.danger_zone_margin = danger_zone_margin
        self._deadzone_m = deadzone_m
        self._track_orientation = track_orientation

        # Camera streaming config
        self._stream_camera = stream_camera
        self._camera_key = camera_key
        self._stream_interval = 1.0 / max(stream_fps, 1.0)
        self._stream_quality = stream_quality
        self._camera_frame: Optional[np.ndarray] = None
        self._last_stream_time = 0.0

        if default_orientation_wxyz is not None:
            self._default_wxyz = np.array(default_orientation_wxyz, dtype=np.float64)
        else:
            self._default_wxyz = np.array([-0.5, -0.5, -0.5, -0.5])

        # Load URDF and build PyRoki robot for IK + FK
        current_path = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(
            current_path, "..", "..", "..", "dependencies", "i2rt", "i2rt", "robot_models", "yam", "yam.urdf"
        )
        mesh_dir = os.path.join(
            current_path, "..", "..", "..", "dependencies", "i2rt", "i2rt", "robot_models", "yam", "assets"
        )
        self._urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_dir)
        self._pk_robot = pk.Robot.from_urdf(self._urdf)
        self._target_link = "link_6"
        self._target_link_index = self._pk_robot.links.names.index(self._target_link)
        self._n_joints = 6

        # Per-arm state
        sides = ["left", "right"] if bimanual else ["left"]
        self._sides = sides
        self._joints: Dict[str, np.ndarray] = {s: np.zeros(self._n_joints) for s in sides}
        self._gripper: Dict[str, float] = {s: _GRIPPER_OPEN for s in sides}
        self._pos_filter: Dict[str, _CriticallyDampedFilter] = {
            s: _CriticallyDampedFilter(omega=smoothing_omega, dim=3) for s in sides
        }
        self._smoothed_target_wxyz: Dict[str, Optional[np.ndarray]] = {s: None for s in sides}
        self._last_act_time = time.time()

        # Delta-control anchors: set when trigger is first pressed
        self._anchor_ctrl_pos: Dict[str, Optional[np.ndarray]] = {s: None for s in sides}
        self._anchor_ee_pos: Dict[str, Optional[np.ndarray]] = {s: None for s in sides}
        self._anchor_ctrl_wxyz: Dict[str, Optional[np.ndarray]] = {s: None for s in sides}
        self._anchor_ee_wxyz: Dict[str, Optional[np.ndarray]] = {s: None for s in sides}
        self._trigger_prev: Dict[str, bool] = {s: False for s in sides}

        # Thread-safe controller state buffer written by Vuer async handler
        self._lock = threading.Lock()
        self._ctrl_state: Dict[str, Any] = {}

        # Warm up JAX / PyRoki JIT (first calls trigger compilation)
        self._compute_ee_position(np.zeros(self._n_joints))
        _solve_ik_teleop(
            self._pk_robot,
            jnp.array(self._target_link_index),
            jnp.array(self._default_wxyz, dtype=jnp.float32),
            jnp.array([0.1, 0.0, 0.15], dtype=jnp.float32),
            jnp.zeros(self._n_joints, dtype=jnp.float32),
        )

        # Start Vuer server in background thread
        self._vuer_thread = threading.Thread(target=self._run_vuer, daemon=True)
        self._vuer_thread.start()
        logger.info(
            "QuestVRAgent started (delta control). Vuer WebXR on port %d. "
            "Squeeze trigger to engage, move hand to control EE.",
            vuer_port,
        )

    # ------------------------------------------------------------------ #
    # Forward kinematics helper
    # ------------------------------------------------------------------ #

    def _compute_ee_pose(self, joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position [x,y,z], quaternion wxyz) for the given joints."""
        Ts = self._pk_robot.forward_kinematics(jnp.array(joints, dtype=jnp.float32))
        ee_pose = jaxlie.SE3(Ts[self._target_link_index])
        pos = np.array(ee_pose.translation())
        wxyz = np.array(ee_pose.rotation().wxyz)
        return pos, wxyz

    def _compute_ee_position(self, joints: np.ndarray) -> np.ndarray:
        """Return the end-effector position [x, y, z] for the given joints."""
        return self._compute_ee_pose(joints)[0]

    # ------------------------------------------------------------------ #
    # Vuer async server (runs in background thread)
    # ------------------------------------------------------------------ #

    def _run_vuer(self) -> None:
        """Start the Vuer WebXR server in a new asyncio event loop."""
        from vuer import Vuer, VuerSession
        from vuer.schemas import ImageBackground, MotionControllers

        app = Vuer(port=self.vuer_port)

        @app.add_handler("CONTROLLER_MOVE")
        async def _on_controller_move(event: Any, _session: Any) -> None:
            with self._lock:
                self._ctrl_state = event.value

        @app.spawn(start=True)
        async def _main(session: VuerSession) -> None:
            session.upsert @ MotionControllers(
                stream=True,
                key="quest-controllers",
                left=True,
                right=True,
            )
            while True:
                frame = None
                if self._stream_camera:
                    with self._lock:
                        if self._camera_frame is not None:
                            frame = self._camera_frame
                            self._camera_frame = None
                if frame is not None:
                    h, w = frame.shape[:2]
                    session.upsert(
                        ImageBackground(
                            frame,
                            aspect=w / h,
                            height=1,
                            distanceToCamera=2,
                            layers=3,
                            format="jpeg",
                            quality=self._stream_quality,
                            key="sim-camera",
                            interpolate=True,
                        ),
                        to="bgChildren",
                    )
                await asyncio.sleep(1.0 / 60)

    # ------------------------------------------------------------------ #
    # Agent protocol
    # ------------------------------------------------------------------ #

    def act(self, obs: Dict[str, Any]) -> Any:
        now = time.time()
        dt = now - self._last_act_time
        self._last_act_time = now

        with self._lock:
            ctrl = dict(self._ctrl_state)

        action: Dict[str, Dict[str, np.ndarray]] = {}

        for side in self._sides:
            ctrl_key = side                 # "left" or "right"
            state_key = f"{side}State"      # "leftState" or "rightState"

            mat16 = ctrl.get(ctrl_key)
            buttons = ctrl.get(state_key, {})

            trigger_pressed = bool(buttons.get("trigger", False))
            trigger_just_pressed = trigger_pressed and not self._trigger_prev[side]
            trigger_just_released = not trigger_pressed and self._trigger_prev[side]
            self._trigger_prev[side] = trigger_pressed

            if mat16 is not None and len(mat16) == 16:
                raw_pos, _ = _extract_pos_quat_from_col_major(mat16)
                ctrl_pos_robot = _webxr_to_robot_pos(raw_pos)
                ctrl_wxyz_robot = _webxr_mat_to_robot_quat(mat16)

                if trigger_just_pressed:
                    self._anchor_ctrl_pos[side] = ctrl_pos_robot.copy()
                    ee_pos, ee_wxyz = self._compute_ee_pose(self._joints[side])
                    self._anchor_ee_pos[side] = ee_pos
                    self._pos_filter[side].reset(ee_pos)
                    if self._track_orientation:
                        self._anchor_ctrl_wxyz[side] = ctrl_wxyz_robot.copy()
                        self._anchor_ee_wxyz[side] = ee_wxyz.copy()
                        self._smoothed_target_wxyz[side] = ee_wxyz.copy()
                    logger.info(
                        "%s trigger pressed — anchor EE at %s",
                        side, self._anchor_ee_pos[side],
                    )

                if trigger_pressed and self._anchor_ctrl_pos[side] is not None:
                    raw_delta = (ctrl_pos_robot - self._anchor_ctrl_pos[side]) * self.position_scale
                    delta = _deadzone(raw_delta, self._deadzone_m)
                    target_pos = self._anchor_ee_pos[side] + delta

                    ik_pos = self._pos_filter[side].step(target_pos, dt)
                    ik_wxyz = self._default_wxyz

                    if self._track_orientation and self._anchor_ctrl_wxyz[side] is not None:
                        delta_q = _quat_mul_wxyz(
                            ctrl_wxyz_robot,
                            _quat_conj_wxyz(self._anchor_ctrl_wxyz[side]),
                        )
                        delta_q = np.array(delta_q)
                        delta_q[2] = -delta_q[2]
                        target_wxyz = _quat_mul_wxyz(delta_q, self._anchor_ee_wxyz[side])
                        target_wxyz /= np.linalg.norm(target_wxyz)

                        a = self.smoothing_alpha
                        if np.dot(target_wxyz, self._smoothed_target_wxyz[side]) < 0:
                            target_wxyz = -target_wxyz
                        self._smoothed_target_wxyz[side] = (
                            a * target_wxyz + (1 - a) * self._smoothed_target_wxyz[side]
                        )
                        norm = np.linalg.norm(self._smoothed_target_wxyz[side])
                        if norm > 0:
                            self._smoothed_target_wxyz[side] /= norm
                        ik_wxyz = self._smoothed_target_wxyz[side]

                    try:
                        raw_joints = np.array(_solve_ik_teleop(
                            self._pk_robot,
                            jnp.array(self._target_link_index),
                            jnp.array(ik_wxyz, dtype=jnp.float32),
                            jnp.array(ik_pos, dtype=jnp.float32),
                            jnp.array(self._joints[side], dtype=jnp.float32),
                        ))
                        raw_joints = _clamp_to_limits(
                            raw_joints, _YAM_JOINT_LIMITS, self.danger_zone_margin,
                        )
                        raw_joints = _limit_joint_velocity(
                            raw_joints, self._joints[side], dt, self.max_joint_vel,
                        )
                        self._joints[side] = raw_joints
                    except Exception:
                        logger.warning(
                            "IK solve failed for %s arm, holding previous joints", side,
                        )

            if trigger_just_released:
                self._anchor_ctrl_pos[side] = None
                self._anchor_ee_pos[side] = None
                self._anchor_ctrl_wxyz[side] = None
                self._anchor_ee_wxyz[side] = None
                self._smoothed_target_wxyz[side] = None

            # Gripper: A/X button closes, B/Y button opens
            a_pressed = buttons.get("aButton", False)
            b_pressed = buttons.get("bButton", False)
            if a_pressed:
                self._gripper[side] = _GRIPPER_CLOSED
            elif b_pressed:
                self._gripper[side] = _GRIPPER_OPEN

            action[side] = {
                "pos": np.concatenate([
                    np.flip(self._joints[side]),
                    [self._gripper[side]],
                ]).astype(np.float32),
            }

        # Stream camera frame to the Quest headset (throttled)
        if self._stream_camera:
            now_stream = time.time()
            if now_stream - self._last_stream_time >= self._stream_interval:
                rgb_dict = obs_get_rgb(obs)
                if rgb_dict:
                    if self._camera_key and self._camera_key in rgb_dict:
                        frame = rgb_dict[self._camera_key]
                    else:
                        frame = next(iter(rgb_dict.values()))
                    with self._lock:
                        self._camera_frame = frame
                    self._last_stream_time = now_stream

        # In combined mode, merge left+right into a single 14-DOF vector
        # under one robot key (for YamPickRedCubeSimRobot with right_arm_only=False)
        if self.bimanual and self.bimanual_combined_key is not None:
            left_pos = action.get("left", {}).get("pos", np.zeros(7, dtype=np.float32))
            right_pos = action.get("right", {}).get("pos", np.zeros(7, dtype=np.float32))
            return {
                self.bimanual_combined_key: {
                    "pos": np.concatenate([left_pos, right_pos]),
                }
            }

        return action

    @remote(serialization_needed=True)
    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        if self.bimanual and self.bimanual_combined_key is not None:
            return {
                self.bimanual_combined_key: {"pos": Array(shape=(14,), dtype=np.float32)},
            }
        spec: Dict[str, Dict[str, Array]] = {
            "left": {"pos": Array(shape=(7,), dtype=np.float32)},
        }
        if self.bimanual:
            spec["right"] = {"pos": Array(shape=(7,), dtype=np.float32)}
        return spec

    def close(self) -> None:
        logger.info("QuestVRAgent shutting down.")
