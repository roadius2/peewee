import os

from peewee_decide.router import DEFAULT_MODELS, _local_models, normalise_name


def test_local_models_parse_pairs():
    assert _local_models("newgle-rel-v1=/srv/m, bad, =x, Other = ~/m2") == {
        "newgle-rel-v1": ("/srv/m", None), "other": (os.path.expanduser("~/m2"), None)}
    assert _local_models("") == {}


def test_registered_names_resolve(monkeypatch):
    monkeypatch.setitem(DEFAULT_MODELS, "newgle-rel-v1", ("/srv/m", None))
    assert normalise_name("newgle-rel-v1") == "newgle-rel-v1"
