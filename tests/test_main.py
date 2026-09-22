"""The `laya` command dispatches every subcommand."""
import pytest

from laya.__main__ import main


@pytest.mark.parametrize("cmd", ["train", "eval", "prepare-data"])
def test_subcommands_are_dispatched(cmd):
    with pytest.raises(SystemExit) as e:
        main([cmd, "--help"])
    assert e.value.code == 0


def test_unknown_command_prints_usage(capsys):
    assert main(["nope"]) == 2
    assert "laya train" in capsys.readouterr().err
