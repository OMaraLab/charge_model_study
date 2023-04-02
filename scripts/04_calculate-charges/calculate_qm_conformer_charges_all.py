#!/usr/bin/env python3

import click
import dataclasses
import os
import numpy as np
import pathlib
import psiresp
import tqdm
from typing import Tuple, Literal

from rdkit import Chem
import qcelemental as qcel
from charge_model_study.filenames import Filenames
from openff.toolkit.topology.molecule import Molecule, unit
from psiresp.constraint import ESPSurfaceConstraintMatrix

from charge_model_study.symmetries import AtomSymmetries
from charge_model_study.qm import ChargeMethod, QMMethod

DIRECTORY = pathlib.Path(__file__).parent.resolve()
DEFAULT_INPUT_ROOT_QM_DIRECTORY = str(
    DIRECTORY.parent / "03_generate-esps"
)


@dataclasses.dataclass
class RespCalculator:
    component_number: int
    conformer_number: int
    charge_method_name: Literal["resp", "resp2", "atb"]
    environment: Literal["solvated", "vacuum"]
    input_root_qm_directory: str = DEFAULT_INPUT_ROOT_QM_DIRECTORY
    output_directory: str = DIRECTORY

    def __post_init__(self):
        self.filenames = Filenames(
            component_number=self.component_number,
            conformer_number=self.conformer_number,
            charge_method_name=self.charge_method_name,
            environment=self.environment,
            root_qm_directory=self.input_root_qm_directory,
        )

        self.charge_method: QMMethod = ChargeMethod[self.charge_method_name.upper(
        )].value
        self.job = self.get_psiresp_job()
        self.offmol = Molecule.from_qcschema(self.job.molecules[0].qcmol)
        self.rdmol = self.offmol.to_rdkit()

    def get_psiresp_job(self):
        orientations = [psiresp.Orientation.parse_file(
            self.filenames.esp_orientation_filename)]
        conformers = [
            psiresp.Conformer(
                qcmol=orientation.qcmol,
                orientations=[orientation],
                is_optimized=True,
            )
            for orientation in orientations
        ]
        molecule = psiresp.Molecule(
            qcmol=conformers[0].qcmol,
            conformers=conformers,
            optimize_geometry=False,
            charge=0,
            multiplicity=1,
        )

        return psiresp.Job(
            molecules=[molecule]
        )

    def write_esp_surface_constraint_matrix(self):
        self.esp_surface_constraint_matrix = self.job.construct_surface_constraint_matrix()
        self.filenames.write_to_file(
            self.filenames.surface_constraint_matrix_conformer_filename,
            self.esp_surface_constraint_matrix.json(),
        )

    def write_atom_symmetries(self):
        symmetric_atoms = list(
            Chem.CanonicalRankAtoms(self.rdmol, breakTies=False))
        self.atom_symmetries = AtomSymmetries.from_openff_molecule(
            self.offmol, symmetric_atoms)
        # self.filenames.write_to_file(
        #     self.filenames.atom_symmetries_filename,
        #     self.atom_symmetries.json(),
        # )

    def get_n_structure_array(self):
        # n_conformers = len(self.conformer_numbers)
        return np.array([1] * len(self.offmol.atoms))

    def generate_respcharges(self):
        mol = self.job.molecules[0]
        dummy = psiresp.ChargeConstraintOptions()
        dummy.add_charge_sum_constraint_for_molecule(
            mol,
            charge=0,
            indices=[0]
        )
        molconstr = psiresp.charge.MoleculeChargeConstraints.from_charge_constraints(
            dummy, molecules=[mol]
        )

        return psiresp.RespCharges(
            restraint_slope=self.charge_method.restraint_slope,
            restrained_fit=self.charge_method.restrain,
            charge_constraints=molconstr,
            surface_constraints=self.esp_surface_constraint_matrix,
        )

    def _generate_sparse_constraint_matrix(self, constraints, targets):
        from psiresp.constraint import SparseGlobalConstraintMatrix

        hydrogen_mask = self.atom_symmetries.get_hydrogen_mask()
        n_structure_array = self.get_n_structure_array()
        combined_constraints, combined_targets = self.atom_symmetries.combine_constraints(
            self.esp_surface_constraint_matrix,
            constraints,
            targets,
        )
        return SparseGlobalConstraintMatrix(
            coefficient_matrix=combined_constraints,
            constant_vector=combined_targets,
            mask=hydrogen_mask,
            n_structure_array=n_structure_array,
        )

    def calculate_charges(self):
        # stage 1
        constraints_1, targets_1 = self.atom_symmetries.generate_stage_1_constraints(
            exclude_methyl_hydrogens=self.charge_method.stage_2
        )
        respcharges_1 = self.generate_respcharges()
        respcharges_1.restraint_height = self.charge_method.restraint_height_stage_1
        respcharges_1._matrix = self._generate_sparse_constraint_matrix(
            constraints_1, targets_1)
        respcharges_1.solve()
        self.filenames.write_to_file(
            self.filenames.stage_1_conformer_filename,
            respcharges_1.json(),
        )

        charges = respcharges_1.charges[0]
        # stage 2
        if self.charge_method.stage_2:
            constraints_2, targets_2 = self.atom_symmetries.generate_stage_2_constraints(
                charges
            )
            respcharges_2 = self.generate_respcharges()
            respcharges_2.restraint_height = self.charge_method.restraint_height_stage_2
            respcharges_2._matrix = self._generate_sparse_constraint_matrix(
                constraints_2, targets_2)
            respcharges_2.solve()
            self.filenames.write_to_file(
                self.filenames.stage_2_conformer_filename,
                respcharges_2.json(),
            )

            charges = respcharges_2.charges[0]

        charges = self.set_and_return_charges(self.offmol, charges)
        pathlib.Path(self.filenames.resp_charges_filename).parent.mkdir(
            parents=True, exist_ok=True)
        np.savetxt(str(self.filenames.resp_charges_filename), charges)
        print(f"Saved charges to {self.filenames.resp_charges_filename}")

        rounded = np.round(charges, 4)
        rounded = self.set_and_return_charges(self.offmol, rounded)
        rounded = np.round(rounded, 4)

        np.savetxt(str(self.filenames.resp_charges_short_filename), rounded)
        print(f"Saved charges to {self.filenames.resp_charges_short_filename}")

    @staticmethod
    def set_and_return_charges(offmol, charges):
        offmol._partial_charges = charges * unit.elementary_charge
        offmol._normalize_partial_charges()
        charges = np.array(
            [float(x/unit.elementary_charge)
             for x in offmol.partial_charges]
        )
        return charges

    def run(self):
        self.write_atom_symmetries()
        self.write_esp_surface_constraint_matrix()
        self.calculate_charges()


@click.command()
@click.option(
    "--mapping-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--input-root-qm-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=DEFAULT_INPUT_ROOT_QM_DIRECTORY,
)
@click.option(
    "--output-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=DIRECTORY,
)
def run_all(
    mapping_file,
    input_root_qm_directory,
    output_directory,
):
    import pandas as pd

    input_root_qm_directory = os.path.abspath(input_root_qm_directory)
    output_directory = os.path.abspath(output_directory)

    df = pd.read_csv(mapping_file)
    for component_number in tqdm.tqdm(sorted(df.number.unique())):
        n_confs = df[df.number == component_number].n_conformers.values[0]
        for conformer_number in tqdm.tqdm(range(1, n_confs + 1), total=n_confs):
            for charge_method_name, environment in [
                ("resp", "vacuum"),
                ("resp2", "vacuum"),
                ("resp2", "solvated")
            ]:
                print(
                    f"Running {component_number} {conformer_number} {charge_method_name} {environment}")
                try:
                    calculator = RespCalculator(
                        component_number=component_number,
                        conformer_number=conformer_number,
                        charge_method_name=charge_method_name,
                        environment=environment,
                        input_root_qm_directory=input_root_qm_directory,
                        output_directory=output_directory,
                    )
                    calculator.run()
                except BaseException as e:
                    print(f"Error: {e}")


if __name__ == "__main__":
    run_all()
