"""Isaac Lab scene for the ToyBlocks backend, built without ToyBlocks' PDDL codegen.

ToyBlocks' ``BaseEnv`` adds the robot (``robot_id``), the scene preset's assets,
a ground plane at z = -1.05, and then calls ``_setup_scene_objects`` for the
blocks. This module registers a preset holding the OpenUp-sized
``workspace_table`` (vecgen requires that exact name) and the staging pads, and
spawns the piece pool as ``block_*`` cuboids. The env is constructed directly;
a ``DirectRLEnv`` is its own ``unwrapped``, which is all ``VecSim`` needs.

Import only after ``app.launch()``.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from toyblocks.scenes import SCENE_PRESETS
from toyblocks.scenes.base import BaseSceneConfig
from toyblocks.tasks.base_env import BaseEnv
from toyblocks.tasks.base_env_cfg import BaseEnvCfg
from toyblocks.utils import rigid_objects

from toyblocks_backend.props import DYNAMIC_FRICTION, STATIC_FRICTION, color_rgb

PRESET_ID = "openup_cube_stack"
ROBOT_ID = "fr3_robotiq_gripper_sysid"  # ToyBlocks' default template robot (FR3 + Robotiq 2F-85)


def _static_box(box):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + box.name,
        init_state=RigidObjectCfg.InitialStateCfg(pos=box.center, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=box.size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=box.color),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=STATIC_FRICTION, dynamic_friction=DYNAMIC_FRICTION),
        ),
    )


@configclass
class OpenUpCubeEnvCfg(BaseEnvCfg):
    episode_length_s = 99999.0
    scene_preset_id = PRESET_ID
    robot_id = ROBOT_ID
    pool: tuple = ()  # props.PoolBody entries


class OpenUpCubeEnv(BaseEnv):
    cfg: OpenUpCubeEnvCfg

    def _setup_scene_objects(self):
        for body in self.cfg.pool:
            self.scene.rigid_objects[body.key] = rigid_objects.create_cuboid(
                prim_path=f"/World/envs/env_.*/{body.key}",
                size=body.size,
                diffuse_color=color_rgb(body.color),
                mass=body.mass,
                init_pos=body.park_position,
                init_quat=(1.0, 0.0, 0.0, 0.0),
                metallic=0.2,
                static_friction=STATIC_FRICTION,
                dynamic_friction=DYNAMIC_FRICTION,
            )


def build_env(table, pads, pool, device="cuda:0"):
    """Build the single-environment scene (``make_env``'s settings, minus gym registration)."""
    SCENE_PRESETS[PRESET_ID] = BaseSceneConfig(
        asset_cfgs={box.name: _static_box(box) for box in (table, *pads)},
        camera_cfgs={},
    )
    cfg = OpenUpCubeEnvCfg()
    cfg.pool = tuple(pool)
    cfg.scene.num_envs = 1
    cfg.sim.device = device
    cfg.sim.use_fabric = True
    cfg.always_spawn_cameras = False
    cfg.enable_visual_markers = False
    return OpenUpCubeEnv(cfg)
