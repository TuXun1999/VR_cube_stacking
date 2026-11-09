"""A one-command-at-a-time vecgen engine for live robot commands.

Mirrors ``_EvalSim`` and ``_OracleEngine`` in ``toyblocks/eval/ll_oracle.py`` at
the pinned ToyBlocks commit: each command is a one-skill ``Program`` that runs
once (no retries, snapshot restores or re-planning), and no episode is ever
reset. Unlike the oracle, this engine steps the simulation itself.

Import only after Isaac Sim is running (``scene.launch``).
"""
from toyblocks.vecgen.engine import Engine, EngineConfig
from toyblocks.vecgen.program import Phase, Program, SkillCall
from toyblocks.vecgen.sim import VecSim

PROFILE = "medium"  # the datagen stiffness profile vecgen's skills are tuned for
ENV = 0  # the live backend always runs a single environment


class LiveSim(VecSim):
    """VecSim with no recorded reset states: the live session sets every pose."""

    def num_reset_states(self) -> int:
        return 0

    def env_slots(self) -> list[int]:
        return [0] * self.num_envs

    def reset_states_by_slot(self) -> dict[int, list[int]]:
        return {0: []}


class LiveEngine(Engine):
    def __init__(self, sim, geo, skills):
        cfg = EngineConfig(
            max_attempts=1, max_replans=0, profile=PROFILE, async_workers=0,
            async_plans=False, async_programs=False, plan_gather_ticks=0,
        )
        super().__init__(sim, geo, skills, program_fn=lambda _env, _rid: None, cfg=cfg)
        self.results = {}
        self._finished = None

    # -- one command ---------------------------------------------------------------

    def start(self, env, call, params):
        st = self.states[env]
        st.assign(Program([call], "live", -1), self.next_episode)
        self.next_episode += 1
        self.phase_idx[env] = 0
        self._begun.discard(env)
        self._wait_since.pop(env, None)
        self.ctx.params[env] = dict(params)
        self.episode_ticks[env] = 0

    def cancel(self, env=ENV):
        """Drop ``env``'s command (a session reset): no segment, no outcome."""
        self.table.cancel([env])
        self.states[env].clear()
        self.snapshots.pop(env, None)
        self._begun.discard(env)
        self._wait_since.pop(env, None)
        self.results.pop(env, None)
        self.ctx.params[env] = {}

    def busy(self, env=ENV):
        return self.states[env].phase is not Phase.IDLE

    def tick(self):
        """One control step: arm targets, one physics step, then phase bookkeeping."""
        self.control()
        self.sim.step()
        self.after_step()

    def run_call(self, name, args, params, on_tick, max_ticks=6000):
        """Run one skill to completion and return ``(success, reason)``.

        ``on_tick`` runs after every physics step; it pumps the OpenUp connection
        and may raise ``ResetRequested``, in which case the caller cancels.
        """
        self.results.pop(ENV, None)
        self.start(ENV, SkillCall(name, tuple(args)), params)
        for _ in range(max_ticks):
            self.tick()
            on_tick()
            if ENV in self.results:
                return self.results.pop(ENV)
        self.cancel(ENV)
        return False, f"{name} did not finish within {max_ticks} control steps"

    # -- the oracle's split tick ---------------------------------------------------------

    def control(self):
        pos, vel, acc, commanded, finished = self.table.gather(
            self.sim.joint_pos() if self.table.needs_joint_pos() else None)
        ids = self._ids[commanded]
        if ids.numel():
            ff = self.table.ff[ids]
            fc = self.cfg.friction_comp
            if self.cfg.target_ff and bool(ff.any()):
                self.sim.command_arm(pos[ids][ff], ids[ff], vel[ids][ff], acc[ids][ff], True, friction_comp=fc)
                if bool((~ff).any()):
                    self.sim.command_arm(pos[ids][~ff], ids[~ff], target_ff=False, friction_comp=fc)
            else:
                self.sim.command_arm(pos[ids], ids, target_ff=False, friction_comp=fc)
            self.sim.flush()
        self.episode_ticks += commanded.long()
        self._finished = finished

    def after_step(self):
        if self._finished is not None:
            done_ids = self._ids[self._finished]
            self._finished = None
            if done_ids.numel():
                self._on_segments_finished(done_ids)
        self._dispatch_ready()
        self.ticks += 1

    # -- episode handling replaced by one outcome per command ----------------------------

    def _assign_idle(self):
        return  # commands arrive through start()

    def _end_episode(self, e, success, reason):
        if self.states[e].program is None:
            return
        self.results[e] = (success, reason)
        self.table.cancel([e])
        self.snapshots.pop(e, None)
        self._begun.discard(e)
        self._wait_since.pop(e, None)
        self.states[e].clear()

    def _retry_or_fail(self, e, reason):
        self._end_episode(e, False, reason)

    def _install(self, segs, owner):
        super()._install({e: s for e, s in segs.items() if self.states[e].program is not None}, owner)
