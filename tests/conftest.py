import pathlib

import pytest

from pycosa.modeling import VariabilityModel

DATA = pathlib.Path(__file__).resolve().parent.parent / "_test_data"


@pytest.fixture(scope="session")
def data_dir():
    return DATA


@pytest.fixture(params=["h2", "sqlite"])
def vm(request):
    return VariabilityModel.from_dimacs_file(
        str(DATA / "feature_models" / f"{request.param}.dimacs")
    )


@pytest.fixture
def h2():
    return VariabilityModel.from_dimacs_file(str(DATA / "feature_models" / "h2.dimacs"))


def all_valid(vm, df):
    return all(vm.is_valid(row.to_dict()) for _, row in df.iterrows())
