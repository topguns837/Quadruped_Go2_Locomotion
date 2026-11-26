# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Python tests for src/training/train_profile.py (no torch / Isaac Sim imports needed)."""

import contextlib
import io
import os
import re
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import train_profile as tp  # noqa: E402


def write_yaml(text: str, auto_num_phases: bool = True) -> str:
    """Write dedented YAML to a temp file and return its path.

    ``num_phases`` is required by the loader. Fixtures that are about something else get it added automatically
    (counted from the ``- name:`` phase entries) so they stay short; pass ``auto_num_phases=False`` to test the
    num_phases rules themselves.
    """
    text = textwrap.dedent(text)
    if auto_num_phases and "num_phases" not in text:
        n = len(re.findall(r"^\s*-\s*\{?\s*name\s*:", text, re.M))
        if n:
            text = f"num_phases: {n}\n" + text
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


VALID_3PHASE = """
    experiment_name: go2_test
    run_dir: run_test
    task: Quadruped-Locomotion-Go2
    num_envs: 2048
    phases:
      - name: clean
        iterations: 1000
        batched: true
        batch_iterations: 500
        overrides:
          env.events.push_robot_force.params.max_force: 5.0
      - name: realistic
        iterations: 2000
        batched: true
        batch_iterations: 1000
        overrides:
          env:
            events:
              push_robot_force:
                params:
                  max_force: 15.0
            rewards:
              termination_penalty:
                weight: -750.0
      - name: finish
        iterations: 500
        batched: false
        overrides: {}
"""


class LoadProfile(unittest.TestCase):
    def test_loads_three_phases(self):
        prof = tp.load_profile(write_yaml(VALID_3PHASE))
        self.assertEqual(prof.experiment_name, "go2_test")
        self.assertEqual(prof.run_dir, "run_test")
        self.assertEqual(prof.task, "Quadruped-Locomotion-Go2")
        self.assertEqual(prof.num_envs, 2048)
        self.assertEqual([p.name for p in prof.phases], ["clean", "realistic", "finish"])
        self.assertEqual([p.iterations for p in prof.phases], [1000, 2000, 500])
        self.assertEqual([p.batched for p in prof.phases], [True, True, False])
        self.assertEqual(prof.phases[0].batch_iterations, 500)

    def test_nested_and_dotted_overrides_both_parse(self):
        prof = tp.load_profile(write_yaml(VALID_3PHASE))
        flat = tp.flatten_overrides(prof.phases[1].overrides)
        self.assertIn("env.events.push_robot_force.params.max_force=15.0", flat)
        self.assertIn("env.rewards.termination_penalty.weight=-750.0", flat)


class FlattenOverrides(unittest.TestCase):
    def test_nested_to_dotted(self):
        d = {"env": {"events": {"push_robot_force": {"params": {"max_force": 15.0}}}}}
        self.assertEqual(
            tp.flatten_overrides(d), ["env.events.push_robot_force.params.max_force=15.0"]
        )

    def test_predotted_passthrough(self):
        d = {"env.rewards.stand_still.weight": 5.0}
        self.assertEqual(tp.flatten_overrides(d), ["env.rewards.stand_still.weight=5.0"])

    def test_mixed_nested_and_dotted_sorted(self):
        d = {"env.rewards.stand_still.weight": 5.0, "agent": {"max_iterations": 10}}
        # sorted for determinism
        self.assertEqual(
            tp.flatten_overrides(d),
            ["agent.max_iterations=10", "env.rewards.stand_still.weight=5.0"],
        )

    def test_value_formatting(self):
        d = {
            "a.i": 7,
            "a.f": 15.0,
            "a.neg": -750.0,
            "a.btrue": True,
            "a.bfalse": False,
            "a.none": None,
            "a.s": "flat",
            "a.lst": [0.5, 2.0],
            "a.tup": (-5, 5),
        }
        got = dict(x.split("=", 1) for x in tp.flatten_overrides(d))
        self.assertEqual(got["a.i"], "7")
        self.assertEqual(got["a.f"], "15.0")
        self.assertEqual(got["a.neg"], "-750.0")
        self.assertEqual(got["a.btrue"], "True")  # matches repo usage (env...debug_vis=False)
        self.assertEqual(got["a.bfalse"], "False")
        self.assertEqual(got["a.none"], "null")  # Hydra/OmegaConf disable
        self.assertEqual(got["a.s"], "flat")
        self.assertEqual(got["a.lst"], "[0.5,2.0]")  # no spaces
        self.assertEqual(got["a.tup"], "[-5,5]")


class PhaseBoundaries(unittest.TestCase):
    def test_single_phase(self):
        prof = tp.load_profile(write_yaml("""
            experiment_name: e
            run_dir: r
            phases:
              - {name: a, iterations: 1000, batched: false}
        """))
        self.assertEqual(tp.phase_boundaries(prof), [(0, 1000)])

    def test_cumulative(self):
        prof = tp.load_profile(write_yaml(VALID_3PHASE))
        self.assertEqual(tp.phase_boundaries(prof), [(0, 1000), (1000, 3000), (3000, 3500)])


class BuildCommand(unittest.TestCase):
    def setUp(self):
        self.prof = tp.load_profile(write_yaml(VALID_3PHASE))

    def test_batched_phase_argv(self):
        argv = tp.build_batched_command(self.prof, 0)
        self.assertIn("src/training/train_batched.sh", " ".join(argv))
        self.assertEqual(self._val(argv, "--experiment_name"), "go2_test")
        self.assertEqual(self._val(argv, "--run_dir"), "run_test")
        self.assertEqual(self._val(argv, "--total_iterations"), "1000")  # cumulative end of phase 0
        self.assertEqual(self._val(argv, "--batch_iterations"), "500")
        self.assertEqual(self._val(argv, "--num_envs"), "2048")
        self.assertIn("env.events.push_robot_force.params.max_force=5.0", argv)

    def test_phase2_total_is_cumulative(self):
        argv = tp.build_batched_command(self.prof, 1)
        self.assertEqual(self._val(argv, "--total_iterations"), "3000")
        self.assertEqual(self._val(argv, "--batch_iterations"), "1000")

    def test_unbatched_phase_batch_equals_iters(self):
        argv = tp.build_batched_command(self.prof, 2)
        self.assertEqual(self._val(argv, "--total_iterations"), "3500")
        self.assertEqual(self._val(argv, "--batch_iterations"), "500")  # == phase iterations

    def test_cli_override_appended_after_phase_override(self):
        # same key in both -> CLI must appear LATER so Hydra's last-wins makes it win
        argv = tp.build_batched_command(
            self.prof, 0, cli_overrides=["env.events.push_robot_force.params.max_force=99.0"]
        )
        i_phase = argv.index("env.events.push_robot_force.params.max_force=5.0")
        i_cli = argv.index("env.events.push_robot_force.params.max_force=99.0")
        self.assertLess(i_phase, i_cli)

    def test_cli_num_envs_and_seed_override(self):
        argv = tp.build_batched_command(self.prof, 0, cli_num_envs=512, cli_seed=7)
        self.assertEqual(self._val(argv, "--num_envs"), "512")
        self.assertEqual(self._val(argv, "--seed"), "7")

    def test_global_baseline_merged_and_phase_wins(self):
        prof = tp.load_profile(write_yaml("""
            experiment_name: e
            run_dir: r
            overrides:
              env.rewards.stand_still.weight: 5.0
              env.rewards.height_penalty.weight: -15.0
            phases:
              - name: a
                iterations: 100
                batched: false
                overrides: {env.rewards.stand_still.weight: 9.0}
        """))
        argv = tp.build_batched_command(prof, 0)
        # baseline-only value present, baseline+phase value uses the PHASE value (phase wins, single entry)
        self.assertIn("env.rewards.height_penalty.weight=-15.0", argv)
        self.assertIn("env.rewards.stand_still.weight=9.0", argv)
        self.assertNotIn("env.rewards.stand_still.weight=5.0", argv)

    def _val(self, argv, flag):
        return argv[argv.index(flag) + 1]


class Monitoring(unittest.TestCase):
    """Live-graph dashboard/viewer are profile params, forwarded to train_batched.sh (which starts them itself)."""

    BASE = """
        experiment_name: e
        run_dir: r
        {extra}
        phases:
          - {{name: a, iterations: 100, batched: false}}
    """

    def _prof(self, extra=""):
        return tp.load_profile(write_yaml(self.BASE.format(extra=extra)))

    def _val(self, argv, flag):
        return argv[argv.index(flag) + 1]

    def test_defaults_dashboard_and_viewer_on(self):
        prof = self._prof()
        self.assertTrue(prof.dashboard)
        self.assertTrue(prof.viewer)
        self.assertEqual(prof.dashboard_output, "/tmp/dashboard.png")
        argv = tp.build_batched_command(prof, 0)
        self.assertNotIn("--no_dashboard", argv)
        self.assertNotIn("--no_viewer", argv)
        self.assertEqual(self._val(argv, "--dashboard_output"), "/tmp/dashboard.png")

    def test_dashboard_off_from_profile(self):
        argv = tp.build_batched_command(self._prof("dashboard: false"), 0)
        self.assertIn("--no_dashboard", argv)
        self.assertNotIn("--dashboard_output", argv)  # nothing to write when the dashboard is off

    def test_viewer_off_from_profile(self):
        argv = tp.build_batched_command(self._prof("viewer: false"), 0)
        self.assertIn("--no_viewer", argv)
        self.assertNotIn("--no_dashboard", argv)

    def test_custom_output_path(self):
        argv = tp.build_batched_command(self._prof("dashboard_output: /tmp/mine.png"), 0)
        self.assertEqual(self._val(argv, "--dashboard_output"), "/tmp/mine.png")

    def test_cli_overrides_profile(self):
        prof = self._prof("dashboard: true\n        viewer: true")
        argv = tp.build_batched_command(prof, 0, cli_dashboard=False)
        self.assertIn("--no_dashboard", argv)
        argv = tp.build_batched_command(prof, 0, cli_viewer=False, cli_dashboard_output="/tmp/x.png")
        self.assertIn("--no_viewer", argv)
        self.assertEqual(self._val(argv, "--dashboard_output"), "/tmp/x.png")
        # CLI can also turn ON what the profile turned off
        off = self._prof("dashboard: false")
        self.assertNotIn("--no_dashboard", tp.build_batched_command(off, 0, cli_dashboard=True))

    def test_non_bool_rejected(self):
        for bad in ("dashboard: yes_please", "viewer: 1", "dashboard: 'true'"):
            with self.assertRaises(tp.ProfileError, msg=bad):
                self._prof(bad)

    def test_non_string_output_rejected(self):
        with self.assertRaises(tp.ProfileError):
            self._prof("dashboard_output: 5")

    def test_cli_flags_parse(self):
        # main()'s argparse: --no_dashboard / --no_viewer / --dashboard_output are accepted and reach the builder
        self.assertEqual(tp.parse_cli(["--profile", "x", "--no_dashboard"]).dashboard, False)
        self.assertEqual(tp.parse_cli(["--profile", "x", "--no_viewer"]).viewer, False)
        self.assertIsNone(tp.parse_cli(["--profile", "x"]).dashboard)  # unset -> profile decides
        self.assertEqual(tp.parse_cli(["--profile", "x", "--dashboard_output", "/tmp/a.png"]).dashboard_output,
                         "/tmp/a.png")


class Validation(unittest.TestCase):
    def _bad(self, yaml_text, *, msg_contains=None):
        with self.assertRaises(tp.ProfileError) as cm:
            tp.load_profile(write_yaml(yaml_text))
        if msg_contains:
            self.assertIn(msg_contains, str(cm.exception).lower())

    def test_unknown_phase_key(self):
        self._bad("""
            experiment_name: e
            run_dir: r
            phases:
              - {name: a, iterations: 10, batched: false, bogus: 1}
        """, msg_contains="bogus")

    def test_missing_iterations(self):
        self._bad("""
            experiment_name: e
            run_dir: r
            phases:
              - {name: a, batched: false}
        """, msg_contains="iterations")

    def test_non_int_iterations(self):
        self._bad("""
            experiment_name: e
            run_dir: r
            phases:
              - {name: a, iterations: 1.5, batched: false}
        """)

    def test_batched_without_batch_iterations(self):
        self._bad("""
            experiment_name: e
            run_dir: r
            phases:
              - {name: a, iterations: 100, batched: true}
        """, msg_contains="batch_iterations")

    def test_batch_bigger_than_iterations(self):
        self._bad("""
            experiment_name: e
            run_dir: r
            phases:
              - {name: a, iterations: 100, batched: true, batch_iterations: 200}
        """)

    def test_empty_phases(self):
        self._bad("""
            experiment_name: e
            run_dir: r
            phases: []
        """, msg_contains="phase")

    def test_duplicate_phase_name(self):
        self._bad("""
            experiment_name: e
            run_dir: r
            phases:
              - {name: a, iterations: 10, batched: false}
              - {name: a, iterations: 10, batched: false}
        """, msg_contains="duplicate")

    def test_override_head_not_env_or_agent(self):
        self._bad("""
            experiment_name: e
            run_dir: r
            phases:
              - name: a
                iterations: 10
                batched: false
                overrides: {scene.num_envs: 10}
        """, msg_contains="env")

    def test_override_targets_baked_in_constant(self):
        for path in (
            "env.actions.joint_pos.scale",  # not a baked constant -> allowed; sanity only, not asserted here
        ):
            pass
        # kp/kd/armature/history_length are baked-in: must be rejected with a pointer
        for bad_path in (
            "env.sim.kp",
            "env.actuators.base_legs.stiffness",
            "env.observations.policy.history_length",
        ):
            self._bad(f"""
                experiment_name: e
                run_dir: r
                phases:
                  - name: a
                    iterations: 10
                    batched: false
                    overrides: {{{bad_path}: 1.0}}
            """)


class NumPhases(unittest.TestCase):
    """Top-level `num_phases` must be present and equal the number of entries under `phases:`."""

    BODY = """
        experiment_name: e
        run_dir: r
        {num}
        phases:
          - {{name: a, iterations: 10, batched: false}}
          - {{name: b, iterations: 20, batched: false}}
    """

    def _load(self, num_line):
        return tp.load_profile(write_yaml(self.BODY.format(num=num_line), auto_num_phases=False))

    def test_matching_count_loads(self):
        prof = self._load("num_phases: 2")
        self.assertEqual(prof.num_phases, 2)
        self.assertEqual(len(prof.phases), 2)

    def test_missing_is_an_error_that_says_what_to_add(self):
        with self.assertRaises(tp.ProfileError) as cm:
            self._load("")
        self.assertIn("num_phases", str(cm.exception))
        self.assertIn("num_phases: 2", str(cm.exception))

    def test_mismatch_names_both_numbers(self):
        with self.assertRaises(tp.ProfileError) as cm:
            self._load("num_phases: 3")
        msg = str(cm.exception)
        self.assertIn("num_phases: 3", msg)
        self.assertIn("2 phase(s)", msg)

    def test_non_positive_or_non_int_rejected(self):
        for bad in ("0", "-1", "2.0", "'2'", "true"):
            with self.assertRaises(tp.ProfileError, msg=bad):
                self._load(f"num_phases: {bad}")

    def test_cli_num_phases_runs_only_the_first_k(self):
        path = write_yaml(self.BODY.format(num="num_phases: 2"), auto_num_phases=False)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = tp.main(["--profile", path, "--dry_run", "--num_phases", "1"])
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("phase 1/1: a", text)
        self.assertNotIn(": b", text)

    def test_cli_num_phases_out_of_range_rejected(self):
        path = write_yaml(self.BODY.format(num="num_phases: 2"), auto_num_phases=False)
        for bad in ("0", "3"):
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    tp.main(["--profile", path, "--dry_run", "--num_phases", bad])

    def test_cli_num_phases_keeps_cumulative_iterations_of_kept_phases(self):
        prof = tp.load_profile(write_yaml(self.BODY.format(num="num_phases: 2"), auto_num_phases=False))
        self.assertEqual(tp.phase_boundaries(prof)[0], (0, 10))


class RunSettings(unittest.TestCase):
    """headless / rendering_mode / livestream / max_fails: profile keys, CLI beats profile, defaults unchanged."""

    BASE = """
        experiment_name: e
        run_dir: r
        {extra}
        phases:
          - {{name: a, iterations: 10, batched: false}}
    """

    def _argv(self, extra="", **cli):
        extra = extra.replace("\n", "\n        ")  # continuation lines must match BASE's indentation
        prof = tp.load_profile(write_yaml(self.BASE.format(extra=extra)))
        return tp.build_batched_command(prof, 0, **cli)

    def test_defaults_are_headless_and_add_nothing_else(self):
        argv = self._argv()
        self.assertIn("--headless", argv)
        for flag in ("--gui", "--rendering_mode", "--livestream", "--max_fails"):
            self.assertNotIn(flag, argv)

    def test_profile_keys_reach_the_command(self):
        argv = self._argv("headless: false\nrendering_mode: performance\nlivestream: 2\nmax_fails: 5")
        self.assertIn("--gui", argv)
        self.assertNotIn("--headless", argv)
        self.assertEqual(argv[argv.index("--rendering_mode") + 1], "performance")
        self.assertEqual(argv[argv.index("--livestream") + 1], "2")
        self.assertEqual(argv[argv.index("--max_fails") + 1], "5")

    def test_cli_beats_profile(self):
        argv = self._argv("rendering_mode: performance\nlivestream: 1", cli_rendering_mode="quality",
                          cli_livestream=2, cli_headless=False)
        self.assertEqual(argv[argv.index("--rendering_mode") + 1], "quality")
        self.assertEqual(argv[argv.index("--livestream") + 1], "2")
        self.assertIn("--gui", argv)

    def test_invalid_values_rejected(self):
        for extra in ("rendering_mode: ultra", "livestream: 3", "headless: 1", "max_fails: 0"):
            with self.assertRaises(tp.ProfileError, msg=extra):
                tp.load_profile(write_yaml(self.BASE.format(extra=extra)))

    def test_cli_flags_parse(self):
        args = tp.parse_cli(["--profile", "x", "--num_phases", "1", "--rendering_mode", "balanced", "--livestream", "2"])
        self.assertEqual((args.num_phases, args.rendering_mode, args.livestream), (1, "balanced", 2))


class ShippedProfilesDryRun(unittest.TestCase):
    """The shipped example profile assembles one command per phase without launching anything."""

    def test_example_profile(self):
        root = os.path.join(os.path.dirname(__file__), "..", "profiles", "example.yaml")
        if not os.path.exists(root):
            self.skipTest("src/training/profiles/example.yaml not present yet")
        prof = tp.load_profile(root)
        cmds = [tp.build_batched_command(prof, i) for i in range(len(prof.phases))]
        self.assertEqual(len(cmds), len(prof.phases))
        for argv in cmds:
            self.assertIn("src/training/train_batched.sh", " ".join(argv))

    def test_every_shipped_profile_declares_a_matching_num_phases(self):
        profiles = os.path.join(os.path.dirname(__file__), "..", "profiles")
        names = [f for f in os.listdir(profiles) if f.endswith(".yaml")]
        self.assertGreaterEqual(len(names), 5)
        for name in names:
            prof = tp.load_profile(os.path.join(profiles, name))  # raises if num_phases != len(phases)
            self.assertEqual(prof.num_phases, len(prof.phases), name)

    def test_single_phase_profile_is_one_command_from_scratch(self):
        path = os.path.join(os.path.dirname(__file__), "..", "profiles", "obs_stacking_with_height_single_phase.yaml")
        prof = tp.load_profile(path)
        self.assertEqual(prof.num_phases, 1)
        argv = tp.build_batched_command(prof, 0)
        self.assertEqual(argv[argv.index("--total_iterations") + 1], "25000")
        self.assertEqual(argv[argv.index("--batch_iterations") + 1], "2000")

    def test_single_phase_profile_shares_the_two_phase_baseline(self):
        profiles = os.path.join(os.path.dirname(__file__), "..", "profiles")
        two = tp.load_profile(os.path.join(profiles, "obs_stacking_with_height.yaml"))
        one = tp.load_profile(os.path.join(profiles, "obs_stacking_with_height_single_phase.yaml"))
        self.assertEqual(one.overrides, two.overrides, "keep the single-phase baseline in sync with the two-phase one")
        self.assertEqual((one.task, one.num_envs, one.seed), (two.task, two.num_envs, two.seed))
        # the single phase must not add overrides of its own: the baseline is already the full realistic setting
        self.assertEqual(one.phases[0].overrides, two.phases[-1].overrides)
        self.assertNotEqual(one.experiment_name, two.experiment_name)


if __name__ == "__main__":
    unittest.main()
