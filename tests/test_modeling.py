import z3

from pycosa.modeling import VariabilityModel


def test_negative_literals_in_pair_clauses():
    vm = VariabilityModel()
    vm.from_dimacs([[-1, -2]], {1: "a", 2: "b"})
    s = z3.Solver()
    s.add(vm.get_z3_pair_clauses())
    # both copies must respect the exclusion
    assert s.check(z3.Bool("a-1"), z3.Bool("b-1")) == z3.unsat
    assert s.check(z3.Bool("a-2"), z3.Bool("b-2")) == z3.unsat
    assert s.check(z3.Bool("a-1"), z3.Bool("b-2")) == z3.sat


def test_select_deselect_do_not_leak():
    clauses = [[1, 2]]
    vm = VariabilityModel()
    vm.from_dimacs(clauses, {1: "a", 2: "b"})
    vm.select(["a"])
    vm.deselect("b")
    assert clauses == [[1, 2]]
    assert vm.is_valid({"a": True, "b": False})
    assert not vm.is_valid({"a": False})
    assert not vm.is_valid({"b": True})


def test_shuffle_preserves_semantics(h2):
    before = sorted(sorted(c) for c in h2.dimacs)
    h2.shuffle(seed=3)
    assert sorted(sorted(c) for c in h2.dimacs) == before
    assert h2.is_satisfiable()


def test_core_and_dead(h2):
    core, dead = h2.core_and_dead_features()
    assert "h2" in core
    assert dead == []


def test_unnamed_variables_get_names():
    vm = VariabilityModel()
    vm.from_dimacs([[1, 3]], {1: "a"})
    assert vm.get_binary_features() == ["a", "_x3"]
