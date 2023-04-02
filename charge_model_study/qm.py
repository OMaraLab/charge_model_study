import enum
from typing import NamedTuple


class QMMethod(NamedTuple):
    method: str
    basis: str
    vdw_radii: str
    vdw_point_density: float
    restraint_height_stage_1: float
    restraint_height_stage_2: float
    restraint_slope: float
    restrain: bool
    stage_2: bool


class ChargeMethod(enum.Enum):
    RESP = QMMethod(
        method="hf",
        basis="6-31g*",
        vdw_radii="msk",
        vdw_point_density=1.0,
        restraint_height_stage_1=0.0005,
        restraint_height_stage_2=0.001,
        restraint_slope=0.1,
        restrain=True,
        stage_2=True,
    )
    RESP2 = QMMethod(
        method="pw6b95",
        basis="aug-cc-pV(D+d)Z",
        vdw_radii="bondi",
        vdw_point_density=2.5,
        restraint_height_stage_1=0.0005,
        restraint_height_stage_2=0.001,
        restraint_slope=0.1,
        restrain=True,
        stage_2=True,
    )

    ATB = QMMethod(
        method="b3lyp",
        basis="6-31g*",
        vdw_radii="msk",
        vdw_point_density=1.0,
        restraint_height_stage_1=0.0,
        restraint_height_stage_2=0.0,
        restraint_slope=0.1,
        restrain=False,
        stage_2=False,
    )
