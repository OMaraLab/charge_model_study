import dataclasses
import contextlib
import pathlib
import re

import numpy as np
import pandas as pd
import click
from numpy.testing import assert_allclose

from charge_model_study.filenames import Filenames

DIRECTORY = pathlib.Path(__file__).parent.resolve()


@contextlib.contextmanager
def capture_oechem_warnings():  # pragma: no cover
    from openeye import oechem

    output_stream = oechem.oeosstream()
    oechem.OEThrow.SetOutputStream(output_stream)
    oechem.OEThrow.Clear()

    yield

    oechem.OEThrow.SetOutputStream(oechem.oeerr)


@contextlib.contextmanager
def capture_toolkit_warnings(run: bool = True):  # pragma: no cover
    """A convenience method to capture and discard any warning produced by external
    cheminformatics toolkits excluding the OpenFF toolkit. This should be used with
    extreme caution and is only really intended for use when processing tens of
    thousands of molecules at once."""

    import logging
    import warnings
    from openff.utilities.exceptions import MissingOptionalDependencyError

    if not run:
        yield
        return

    warnings.filterwarnings("ignore")

    toolkit_logger = logging.getLogger("openff.toolkit")
    openff_logger_level = toolkit_logger.getEffectiveLevel()
    toolkit_logger.setLevel(logging.ERROR)

    try:
        with capture_oechem_warnings():
            yield
    except MissingOptionalDependencyError:
        yield

    toolkit_logger.setLevel(openff_logger_level)


def get_new_itp_charges(
    itp_mapped_smiles: str,
    charge_mapped_smiles: str,
    charges: np.ndarray,
):
    from openff.toolkit.topology.molecule import Molecule, unit

    with capture_toolkit_warnings():
        itp_offmol = Molecule.from_mapped_smiles(
            itp_mapped_smiles,
            allow_undefined_stereo=True
        )
        charge_offmol = Molecule.from_mapped_smiles(
            charge_mapped_smiles,
            allow_undefined_stereo=True
        )
        _, itp_to_mapped = Molecule.are_isomorphic(
            itp_offmol,
            charge_offmol,
            return_atom_map=True
        )

        n_charges = len(charges)
        n_itp_mapped_charges = len(itp_to_mapped)
        if n_charges != n_itp_mapped_charges:
            raise AssertionError(
                f"Charges length don't match: given {n_charges} != {n_itp_mapped_charges}")

    mapped_to_itp_ix = [
        itp_to_mapped[itp_idx]
        for itp_idx in sorted(itp_to_mapped)
    ]
    new_itp_charges = charges[mapped_to_itp_ix]
    total_charge = charge_offmol.total_charge
    if not isinstance(total_charge, float):
        total_charge = float(total_charge / unit.elementary_charge)
    offset = (total_charge - new_itp_charges.sum()) / charge_offmol.n_atoms
    new_itp_charges += offset
    return new_itp_charges


def replace_itp_charges(
    input_itp_file: str,
    output_itp_file: str,
    itp_mapped_smiles: str,
    charge_mapped_smiles: str,
    charges: np.ndarray,
    moleculetype: str = "MOL"
):
    import MDAnalysis as mda
    from openff.toolkit.topology.molecule import Molecule, unit

    new_itp_charges = get_new_itp_charges(
        itp_mapped_smiles,
        charge_mapped_smiles,
        charges
    )

    with open(input_itp_file, "r") as f:
        content = f.read()

    before_bonds, after_bonds = content.split("[ bonds ]")
    before_atoms, atom_section = before_bonds.split("[ atoms ]")

    # == replace charges ==
    charge_pattern = (
        "([ ]+\d+[ ]+.+[ ]+\d+[ ]+.+[ ]+.+[ ]+\d+[ ]+)"
        "(?P<charge> [-]?\d\.\d+)"
        "([ ]+\d+\.\d+\n)"
    )
    # replace once with a placeholder to avoid replacing the same line twice
    atom_section = re.sub(charge_pattern, r"\1PLACEHOLDER\3", atom_section)

    placeholder_pattern = r"(.+)(PLACEHOLDER)(.+)"
    for charge in new_itp_charges:
        replacement = r"\1 " + str(charge) + r"\3"
        atom_section = re.sub(placeholder_pattern,
                              replacement, atom_section, count=1)

    # == replace residue name ==
    resname_pattern = (
        "([ ]+\d+[ ]+.+[ ]+1[ ]+)"
        "(?P<resname>[a-zA-Z0-9_-]+)"
        "([ ]+.+[ ]+\d+[ ]+[-]?\d\.\d+[ ]+\d+\.\d+\n)"
    )
    molname_replacement = r"\1" + moleculetype + r"\3"
    atom_section = re.sub(resname_pattern, molname_replacement, atom_section)

    # == replace molecule type ==
    molname_pattern = "(nrexcl[\s]*\n)([a-zA-Z0-9_-]+)(\s+3\s*\n)"
    if not re.search(molname_pattern, before_atoms):
        raise ValueError(f"Could not find moleculetype pattern")
    before_atoms = re.sub(
        molname_pattern,
        molname_replacement,
        before_atoms,
        count=1
    )

    content = before_atoms + "[ atoms ]" + \
        atom_section + "[ bonds ]" + after_bonds
    with open(output_itp_file, "w") as f:
        f.write(content)
    print(f"Wrote {output_itp_file}")

    # == check changes ==
    u2 = mda.Universe(output_itp_file)
    charges = u2.atoms.charges

    assert_allclose(charges, new_itp_charges, atol=1e-4)
    assert u2.residues[0].moltype == moleculetype, f"{u2.residues[0].moltype} != {moleculetype}"
    assert u2.residues[0].resname == moleculetype, f"{u2.residues[0].resname} != {moleculetype}"


def write_openff_top_file(
    output_top_file: str,
    itp_mapped_smiles: str,
    charge_mapped_smiles: str,
    charges: np.ndarray,
    forcefield: str = "openff-2.0.0.offxml",
    moleculetype: str = "MOL"
):
    import MDAnalysis as mda
    from openff.toolkit import ForceField, Molecule
    from openff.toolkit.topology.molecule import unit

    new_itp_charges = get_new_itp_charges(
        itp_mapped_smiles,
        charge_mapped_smiles,
        charges
    )

    with capture_toolkit_warnings():
        offmol = Molecule.from_mapped_smiles(
            itp_mapped_smiles,
            allow_undefined_stereo=True
        )
        offmol._partial_charges = new_itp_charges * unit.elementary_charge
        offmol.name = moleculetype
        for atom in offmol.atoms:
            atom.metadata["residue_number"] = 1
            atom.metadata["residue_name"] = moleculetype

        ff = ForceField(forcefield)
        interchange = ff.create_interchange(
            offmol.to_topology(),
            charge_from_molecules=[offmol]
        )
        interchange.to_top(output_top_file)
    print(f"Wrote {output_top_file}")

    u2 = mda.Universe(output_top_file, topology_format="itp")
    charges = u2.atoms.charges

    assert_allclose(charges, new_itp_charges, atol=1e-4)
    assert u2.residues[0].moltype == moleculetype
    assert u2.residues[0].resname == moleculetype


def write_gro_file(
    input_pdb_file,
    output_gro_file
):
    import MDAnalysis as mda

    u = mda.Universe(input_pdb_file)
    u.atoms.write(output_gro_file)


@dataclasses.dataclass
class FileWriter:

    component_number: int
    conformer_number: int
    root_charge_directory: str
    root_hfe_directory: str
    root_conformer_directory: str
    forcefield: str

    def __post_init__(self):
        self.filenames = Filenames(
            component_number=self.component_number,
            conformer_number=self.conformer_number,
            root_charge_directory=self.root_charge_directory,
            root_conformer_directory=self.root_conformer_directory,
            root_hfe_directory=self.root_hfe_directory,
            forcefield=self.forcefield
        )
        charge_df = pd.read_csv(
            self.filenames.collated_charge_filename,
            index_col=0
        )
        self.charge_df = charge_df[
            (charge_df.component_number == self.component_number) &
            (charge_df.conformer_number == self.conformer_number)
        ]

    def get_charges(
        self,
        charge_method: str,
        environment: str = None,
    ):
        charges = self.charge_df[self.charge_df.charge_method == charge_method]
        if environment is not None:
            charges = charges[charges.environment == environment]
        charges = charges.sort_values(by="atom_index").charge.values
        if len(charges) == 0:
            raise ValueError(f"No charges found for {charge_method}")
        return charges

    def get_mapped_smiles(self):
        itp_smiles_file = self.filenames.atb_lowest_itp_smiles_filename
        charge_smiles_file = self.filenames.smiles_filename

        with open(itp_smiles_file, "r") as f:
            itp_mapped_smiles = f.read().strip()
        with open(charge_smiles_file, "r") as f:
            charge_mapped_smiles = f.read().strip()
        return {
            "itp_mapped_smiles": itp_mapped_smiles,
            "charge_mapped_smiles": charge_mapped_smiles
        }

    def write_openff_files(
        self,
        charge_method: str,
        environment: str = None,
    ):
        charges = self.get_charges(
            charge_method=charge_method,
            environment=environment
        )

        root_output_dir = output_dir = pathlib.Path(
            self.filenames.hfe_directory) / charge_method
        output_dir = root_output_dir / "01_preparation"
        output_dir.mkdir(exist_ok=True, parents=True)
        output_top_file = str(output_dir / "mol.top")
        mapped_smiles = self.get_mapped_smiles()
        forcefield = f"{self.forcefield}.offxml"

        write_openff_top_file(
            output_top_file=output_top_file,
            charges=charges,
            forcefield=forcefield,
            **mapped_smiles
        )

        output_gro_file = str(root_output_dir / "mol.gro")
        write_gro_file(
            self.filenames.atb_lowest_pdb_file,
            output_gro_file
        )
        print(f"Wrote {output_gro_file}")

    def write_atb_files(
        self,
        charge_method: str,
        environment: str = None,
    ):
        charges = self.get_charges(
            charge_method=charge_method,
            environment=environment
        )

        input_itp_file = self.filenames.atb_lowest_itp_file
        output_dir = pathlib.Path(self.filenames.hfe_directory) / charge_method
        output_dir.mkdir(exist_ok=True, parents=True)
        output_itp_file = str(output_dir / "mol.itp")
        mapped_smiles = self.get_mapped_smiles()

        replace_itp_charges(
            input_itp_file=input_itp_file,
            output_itp_file=output_itp_file,
            charges=charges,
            **mapped_smiles
        )

        output_gro_file = str(output_dir / "mol.gro")
        write_gro_file(
            self.filenames.atb_lowest_pdb_file,
            output_gro_file
        )
        print(f"Wrote {output_gro_file}")


@click.group()
def cli():
    pass


@cli.command("write-atb")
@click.option(
    "--component-number",
    type=click.INT,
    required=True,
)
@click.option(
    "--conformer-number",
    type=click.INT,
    required=True,
)
@click.option(
    "--input-root-charge-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
)
@click.option(
    "--input-root-conformer-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
)
@click.option(
    "--output-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=DIRECTORY,
)
@click.option(
    "--charge-method",
    "charge_methods",
    type=str,
    required=True,
    multiple=True
)
def write_atb_files(
    component_number,
    conformer_number,
    input_root_charge_directory,
    input_root_conformer_directory,
    output_directory,
    charge_methods,
):
    writer = FileWriter(
        component_number=component_number,
        conformer_number=conformer_number,
        root_charge_directory=input_root_charge_directory,
        root_conformer_directory=input_root_conformer_directory,
        root_hfe_directory=output_directory,
        forcefield="atb"
    )
    for method in charge_methods:
        writer.write_atb_files(charge_method=method)


@cli.command("write-openff")
@click.option(
    "--component-number",
    type=click.INT,
    required=True,
)
@click.option(
    "--conformer-number",
    type=click.INT,
    required=True,
)
@click.option(
    "--input-root-charge-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
)
@click.option(
    "--input-root-conformer-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
)
@click.option(
    "--output-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=DIRECTORY,
)
@click.option(
    "--forcefield",
    type=str,
    required=True,
)
@click.option(
    "--charge-method",
    "charge_methods",
    type=str,
    required=True,
    multiple=True
)
def write_openff_files(
    component_number,
    conformer_number,
    input_root_charge_directory,
    input_root_conformer_directory,
    output_directory,
    forcefield,
    charge_methods,
):
    writer = FileWriter(
        component_number=component_number,
        conformer_number=conformer_number,
        root_charge_directory=input_root_charge_directory,
        root_conformer_directory=input_root_conformer_directory,
        root_hfe_directory=output_directory,
        forcefield=forcefield
    )
    for method in charge_methods:
        writer.write_openff_files(charge_method=method)


if __name__ == "__main__":
    cli()
