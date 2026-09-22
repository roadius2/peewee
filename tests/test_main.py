"""The `peewee` command: the installed console script, `python -m peewee_decide`, and dispatch."""
import os
import shutil
import subprocess
import sys
import types

import pytest

from peewee_decide.__main__ import main


@pytest.mark.parametrize("cmd", ["serve", "export-onnx", "train", "eval", "prepare-data"])
def test_subcommands_are_dispatched(cmd, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "uvicorn", types.ModuleType("uvicorn"))    # serve imports it; not in [dev]
    with pytest.raises(SystemExit) as e:
        main([cmd, "--help"])
    assert e.value.code == 0
    assert "peewee " + cmd in capsys.readouterr().out


def test_unknown_command_prints_usage(capsys):
    assert main(["nope"]) == 2
    assert "peewee train" in capsys.readouterr().err


def test_console_script_is_registered():
    from importlib.metadata import entry_points
    eps = entry_points()
    scripts = eps.select(group="console_scripts") if hasattr(eps, "select") else eps.get("console_scripts", [])
    assert {e.name: e.value for e in scripts}.get("peewee") == "peewee_decide.__main__:main"


def _launcher(kind):
    if kind == "module":
        return [sys.executable, "-m", "peewee_decide"]
    exe = shutil.which("peewee", path=os.path.dirname(sys.executable))
    if exe is None:
        pytest.skip("the peewee console script is not installed next to this interpreter")
    return [exe]


@pytest.mark.parametrize("kind", ["script", "module"])
def test_peewee_runs_as_a_program(kind):
    cmd = _launcher(kind)
    bare = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    assert bare.returncode == 2 and "peewee train" in bare.stderr
    helped = subprocess.run(cmd + ["eval", "--help"], capture_output=True, text=True, timeout=120, check=False)
    assert helped.returncode == 0 and "peewee eval" in helped.stdout
