#!/usr/bin/env python3

import click
import dataclasses
import numpy as np
import pathlib

import qcelemental as qcel
from charge_model_study.filenames import Filenames
from openff.toolkit.topology.molecule import Molecule, unit
from openff.toolkit.utils.toolkits import OpenEyeToolkitWrapper

DIRECTORY = pathlib.Path(__file__).parent.resolve()
DEFAULT_INPUT_CONFORMER_DIRECTORY = str(
    DIRECTORY.parent / "02_generate-conformers"
)


@dataclasses.dataclass
class ChargeCalculator:
    component_number: int
    conformer_number: int
    input_conformer_directory: str = DEFAULT_INPUT_CONFORMER_DIRECTORY
    output_directory: str = DIRECTORY

    def __post_init__(self):
        self.filenames = Filenames(
            component_number=self.component_number,
            conformer_number=self.conformer_number,
            charge_method_name="",
            environment="",
            root_conformer_directory=self.input_conformer_directory,
            graph_directory=self.output_directory,
        )

    def run(self):
        self.calculate_am1bcc()

    def calculate_am1bcc(self):
        # don't use optimized structure to mimic working conditions
        qcmol = qcel.models.Molecule.parse_file(
            self.filenames.generated_conformer_qcschema_filename)
        offmol = Molecule.from_qcschema(qcmol)
        offmol.assign_partial_charges(
            "am1bcc",
            use_conformers=offmol.conformers,
            toolkit_registry=OpenEyeToolkitWrapper()
        )
        charges = np.array([
            float(x/unit.elementary_charge)
            for x in offmol.partial_charges
        ])

        np.savetxt(charges, self.filenames.am1bcc_charges_filename)
        print(f"Wrote to {self.filenames.am1bcc_charges_filename}")


@click.command()
@click.option(
    "--component_number",
    type=int,
    required=True,
)
@click.option(
    "--conformer_number",
    type=int,
    required=True,
)
@click.option(
    "--input-conformer-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=DEFAULT_INPUT_CONFORMER_DIRECTORY,
)
@click.option(
    "--output-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=DIRECTORY,
)
def main(
        component_number: int,
        conformer_number: int,
        input_conformer_directory: str,
        output_directory: str,
):
    calculator = ChargeCalculator(
        component_number=component_number,
        conformer_number=conformer_number,
        input_conformer_directory=input_conformer_directory,
        output_directory=output_directory,
    )
    calculator.run()


if __name__ == "__main__":
    main()
