# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration test: every override path in the shipped training profiles resolves on the real env/agent cfg.

Needs the Isaac Lab import chain (gym registry + configclasses), so it is skipped on an interpreter that cannot
import it. Run INSIDE the container:
    ../IsaacLab/isaaclab.sh -p -m pytest scripts/tests/test_training_profiles_paths.py
This is the typo-catcher that confirms the recovered historical values (e.g. the `aggressive` magnitudes) map to
fields that actually exist, before a multi-hour run spends GPU time only to have Hydra reject a path.
"""

import glob
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import train_profile as tp  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROFILE_GLOB = os.path.join(REPO_ROOT, "training_profiles", "*.yaml")

# Import the registered env/agent cfgs. Guarded so the pure-Python suite on a plain interpreter just skips this.
try:
    import gymnasium as gym  # noqa: F401
    import quadruped_go2_locomotion.tasks  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry

    _IMPORT_ERR = None
except Exception as e:  # noqa: BLE001 -- torch / pxr / Kit not available here
    _IMPORT_ERR = e


def _resolve(obj, dotted: str):
    """Walk a dotted path over a configclass tree and a params dict; return (ok, where_it_broke)."""
    cur = obj
    walked = []
    for seg in dotted.split("."):
        walked.append(seg)
        if isinstance(cur, dict):
            if seg not in cur:
                return False, ".".join(walked)
            cur = cur[seg]
        elif hasattr(cur, seg):
            cur = getattr(cur, seg)
        else:
            return False, ".".join(walked)
    return True, None


@unittest.skipIf(_IMPORT_ERR is not None, f"Isaac Lab not importable here: {_IMPORT_ERR}")
class ShippedProfilePathsResolve(unittest.TestCase):
    def _cfgs(self, task):
        env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        return {"env": env_cfg, "agent": agent_cfg}

    def test_all_shipped_override_paths_exist(self):
        files = sorted(glob.glob(PROFILE_GLOB))
        self.assertTrue(files, "no training_profiles/*.yaml found")
        failures = []
        for path in files:
            prof = tp.load_profile(path)
            roots = self._cfgs(prof.task)
            maps = [prof.overrides] + [p.overrides for p in prof.phases]
            for ov in maps:
                for dotted in tp._flatten_map(ov):
                    head, _, rest = dotted.partition(".")
                    ok, broke = _resolve(roots[head], rest)
                    if not ok:
                        failures.append(f"{os.path.basename(path)}: {dotted} (missing at {head}.{broke})")
        self.assertEqual(failures, [], "override paths not found on the real cfg:\n" + "\n".join(failures))

    def test_disable_term_is_accepted(self):
        # Setting an event term to None is how a phase disables it (Hydra `...=null`); confirm the field exists.
        roots = self._cfgs(tp.DEFAULT_TASK)
        ok, broke = _resolve(roots["env"], "events.push_robot_force")
        self.assertTrue(ok, f"env.events.push_robot_force missing at {broke}")


if __name__ == "__main__":
    unittest.main()
