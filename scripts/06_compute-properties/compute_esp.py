import dataclasses
import os
import pathlib

import tqdm
import pandas as pd
import click
import qcelemental as qcel
import numpy as np

from charge_model_study.filenames import Filenames
from openff.toolkit.topology import Molecule
from openff.units import unit


def calculate_esp(
    grid_coordinates: unit.Quantity,  # N x 3
    atom_coordinates: unit.Quantity,  # M x 3
    charges: unit.Quantity,  # M
    with_units: bool = False,
) -> unit.Quantity:

    AU_ESP = unit.atomic_unit_of_energy / unit.elementary_charge
    ke = 1 / (4 * np.pi * unit.epsilon_0)

    displacement = grid_coordinates[:, None, :] - \
        atom_coordinates[None, :, :]  # N x M x 3
    distance = (displacement ** 2).sum(axis=-1) ** 0.5  # N x M
    inv_distance = 1 / distance

    esp = ke * (inv_distance @ charges)  # N

    esp_q = esp.m_as(AU_ESP)
    if not with_units:
        return esp_q
    return esp


def calculate_dipole(
    atom_coordinates: unit.Quantity,
    charges: unit.Quantity,
    with_units: bool = False,
) -> np.ndarray:
    # atom_coordinates = atom_coordinates - atom_coordinates.mean(axis=0)
    charges = charges.reshape((-1, 1))

    ucharges = charges.m_as(unit.elementary_charge)
    # atom_coordinates = atom_coordinates - (ucharges * atom_coordinates).mean(axis=0)

    dipole = (atom_coordinates * charges).sum(axis=0)
    dipole_q = dipole.m_as(unit.debye)
    if not with_units:
        return dipole_q
    return dipole


@dataclasses.dataclass
class PropertyCalculator:
    component_number: int
    conformer_number: int
    root_qm_directory: str
    root_charge_directory: str

    def __post_init__(self):
        import pandas as pd

        self.filenames = Filenames(
            component_number=self.component_number,
            conformer_number=self.conformer_number,
            root_qm_directory=self.root_qm_directory,
            charge_method_name="resp",
            environment="vacuum",
            root_charge_directory=self.root_charge_directory
        )
        self.esp = np.loadtxt(self.filenames.esp_esp_filename)
        self.grid = np.loadtxt(self.filenames.esp_grid_filename) * unit.angstrom
        self.qcmol = qcel.models.Molecule.parse_file(
            self.filenames.optimized_conformer_qcschema_filename,
            encoding="json"
        )
        pathlib.Path(self.filenames.conformer_properties_directory).mkdir(
            exist_ok=True, parents=True)
        self.offmol = Molecule.from_qcschema(self.qcmol)
        self.coordinates = self.offmol.conformers[0]

        df = pd.read_csv(self.filenames.collated_charge_filename, index_col=0)
        self.df = df[df.component_number == self.component_number]

    @staticmethod
    def get_charges(
        df,
        charge_method: str,
    ):
        charges = df[df.charge_method == charge_method]
        n_atoms = charges.atom_index.max() + 1
        if len(charges) != n_atoms:
            raise ValueError(
                f"Found {len(charges)} charges for {charge_method}")

        charges = charges.sort_values(by="atom_index").charge.values

        if len(charges) == 0:
            raise ValueError(f"No charges found for {charge_method}")
        return charges * unit.elementary_charge

    def compute_esp(self, charge_method: str, conformer_number: int = 0):
        df = self.df[self.df.conformer_number == conformer_number]
        charges = self.get_charges(df, charge_method)
        esp = calculate_esp(
            grid_coordinates=self.grid,
            atom_coordinates=self.coordinates,
            charges=charges,
            with_units=False,
        )

        filename = os.path.join(
            os.path.abspath(self.filenames.conformer_properties_directory),
            f"esp_{charge_method}_conformer-{conformer_number:02d}.dat"
        )
        np.savetxt(filename, esp)
        print(f"Saved {filename}")
        return esp

    def compute_dipole(self, charge_method: str, conformer_number: int = 0):
        df = self.df[self.df.conformer_number == conformer_number]
        charges = self.get_charges(df, charge_method)

        esp = calculate_dipole(
            atom_coordinates=self.coordinates,
            charges=charges,
            with_units=False,
        )

        filename = os.path.join(
            os.path.abspath(self.filenames.conformer_properties_directory),
            f"dipole_{charge_method}_conformer-{conformer_number:02d}.dat"
        )
        np.savetxt(filename, esp)
        print(f"Saved {filename}")
        return esp

    def run(self):
        df = self.df[self.df.conformer_number == 0]
        for charge_method in df.charge_method.unique():
            if charge_method == "resp2":
                continue
            self.compute_esp(charge_method)
            self.compute_dipole(charge_method)

        self.compute_esp("atb_conformer", conformer_number=1)
        self.compute_dipole("atb_conformer", conformer_number=1)


@click.command()
@click.option(
    "--root-qm-directory",
    type=str,
    required=True,
    help="Root directory containing QM calculations. See charge_model_study.filenames.Filenames.",
)
@click.option(
    "--root-charge-directory",
    type=str,
    required=True,
    help="Root directory containing charges. See charge_model_study.filenames.Filenames.",

)
def run_property_calculator(
    root_qm_directory: str,
    root_charge_directory: str,
):
    fn = Filenames(
        component_number=0,
        conformer_number=0,
        root_charge_directory=root_charge_directory,
    )
    charge_file = fn.collated_charge_filename
    df = pd.read_csv(charge_file, index_col=0)
    for (component_number, conformer_number), _ in tqdm.tqdm(
        df.groupby(by=["component_number", "conformer_number"])
    ):
        if conformer_number == 0:
            continue
        calculator = PropertyCalculator(
            component_number=component_number,
            conformer_number=conformer_number,
            root_qm_directory=root_qm_directory,
            root_charge_directory=root_charge_directory,
        )
        calculator.run()


if __name__ == "__main__":
    run_property_calculator()
