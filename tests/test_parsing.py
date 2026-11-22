import itertools

import pytest

from pycosa.modeling import VariabilityModel
from pycosa.parsing import DimacsParser, SPLCParser


def test_dimacs_parses_test_models(data_dir):
    p = DimacsParser()
    p.parse(str(data_dir / "feature_models" / "h2.dimacs"))
    assert len(p.get_features()) == 19
    assert len(p.get_clauses()) == 22
    assert p.get_feature(7) == "h2"
    assert p.get_index("h2") == 7


def test_dimacs_robustness():
    text = """c a plain comment without index
c 1 A
c 2$ aux
c   3   B with spaces

p cnf 4 3
1 -2
 0
2   3 0 -1 0
"""
    p = DimacsParser()
    p.parse_string(text)
    assert p.get_clauses() == [[1, -2], [2, 3], [-1]]
    assert p.get_feature(2) == "aux"
    assert p.get_feature(3) == "B with spaces"
    # declared but unnamed variable
    assert p.get_feature(4) == "_x4"


def test_unparsed_parser_raises():
    with pytest.raises(RuntimeError):
        DimacsParser().get_feature(1)


def test_unknown_feature_raises(data_dir):
    p = DimacsParser()
    p.parse(str(data_dir / "feature_models" / "h2.dimacs"))
    with pytest.raises(ValueError):
        p.get_index("does-not-exist")


def _valid_configs(parser):
    vm = VariabilityModel()
    vm.from_dimacs(parser.get_clauses(), parser.get_index2features())
    features = vm.get_binary_features()
    valid = set()
    for values in itertools.product([False, True], repeat=len(features)):
        config = dict(zip(features, values))
        if vm.is_valid(config):
            valid.add(frozenset(f for f, v in config.items() if v))
    return valid


def test_splc_parser(data_dir):
    p = SPLCParser()
    p.parse(str(data_dir / "splc" / "toy.xml"))
    assert p.get_features() == [
        "root",
        "Compression",
        "Encryption",
        "Engine",
        "EngineA",
        "EngineB",
        "Logging",
    ]
    assert "threads" in p.numeric_options

    valid = _valid_configs(p)
    for config in valid:
        assert {"root", "Engine"} <= config
        # alternative group: exactly one engine
        assert len(config & {"EngineA", "EngineB"}) == 1
        # implication
        if "Logging" in config:
            assert "Compression" in config
        # cross-tree constraint
        assert not {"Encryption", "EngineB"} <= config
    # A/B x Compression x Logging(->Compression) x Encryption(only with A)
    # EngineA: 3 (comp/log) * 2 (enc) = 6, EngineB: 3 -> 9
    assert len(valid) == 9
