#!/usr/bin/env python3

import click
import dataclasses
import os
from typing import Optional, Set
import pandas as pd
import tqdm
import json
import pathlib

from openff.toolkit.topology import Molecule
from openff.evaluator.datasets import PhysicalPropertyDataSet
from charge_model_study.config import Config
from charge_model_study.filenames import Filenames

DIRECTORY = pathlib.Path(__file__).parent.resolve()
DEFAULT_CONFIG = DIRECTORY.parent / "00_config.yaml"
DEFAULT_INPUT_CSV = DIRECTORY.parent / \
    "01_curate-data" / "subset_freesolv_output.csv"
DEFAULT_OUTPUT_MAPPINGS = DIRECTORY / "mapped_smiles.json"
DEFAULT_OUTPUT_CSV = DIRECTORY / "component_mappings.csv"


@dataclasses.dataclass
class ConformerGenerator:
    """
    Class to generate conformers for a single molecule
    """

    index: int
    mapped_smiles: str
    output_directory: str = DIRECTORY
    n_max_conformers: int = 10
    n_conformer_pool: int = 4000
    elf_percentage: float = 5.0
    conformer_rmsd_threshold: float = 0.05
    config: Config = Config()

    def __post_init__(self):
        self.filenames = Filenames(
            component_number=self.index, root_conformer_directory=self.output_directory)

    @property
    def component_name(self):
        return f"component-{self.index:03d}"

    def run(self) -> int:
        rdmol = self.generate_conformers()
        self.write_conformer_qcschemas(rdmol)
        self.write_conformer_xyzs(rdmol)
        self.write_smiles()
        return rdmol.GetNumConformers()

    def generate_conformers(self):
        from rdkit_utilities import GenerateConformers

        mol = Molecule.from_mapped_smiles(self.mapped_smiles)
        rdmol = mol.to_rdkit()
        GenerateConformers(
            rdmol,
            numConfs=self.n_max_conformers,
            numConfPool=self.n_conformer_pool,
            pruneRmsThresh=0.02,  # angstrom for initial conformer generation
            forcefield="MMFF94",
            optimizeConfs=True,  # optimize conformer pool before doing energy and ELF selection
            energyWindow=10.0,  # kcal/mol for energy selection
            selectELFConfs=True,
            ELFPercentage=self.elf_percentage,  # keep lowest 20% ELF energy conformers
            removeTransAcidConformers=True,  # no trans carboxylic acids
            maximizeDiversity=True,  # select conformers that are diverse by RMSD
            # angstrom for diverse RMSD selection
            diverseRmsThresh=self.conformer_rmsd_threshold,
        )
        print(
            f"Generated {rdmol.GetNumConformers()} conformers for {self.mapped_smiles}")
        return rdmol

    def write_conformer_xyzs(self, rdmol):
        import MDAnalysis as mda

        u = mda.Universe(rdmol)
        for i, ts in enumerate(u.trajectory, 1):
            conformer_config = Filenames(
                component_number=self.index,
                conformer_number=i,
                root_conformer_directory=self.output_directory,
            )
            u.atoms.write(conformer_config.generated_conformer_xyz_filename)

    def write_conformer_qcschemas(self, rdmol):
        from rdkit_utilities.rdDistGeom import CalculateElectrostaticEnergy
        all_energies = CalculateElectrostaticEnergy(rdmol)

        offmol = Molecule.from_rdkit(rdmol)
        for i in range(len(offmol.conformers)):
            qcmol = offmol.to_qcschema(conformer=i)
            conformer_config = Filenames(
                component_number=self.index,
                conformer_number=i + 1,
                root_conformer_directory=self.output_directory,
            )
            qcmol.extras[self.config.qcelemental_conformer_energy_tag] = all_energies[i]
            qcmol.extras[self.config.qcelemental_component_tag] = self.index
            qcmol.extras[self.config.qcelemental_conformer_tag] = conformer_config.conformer_number
            qcmol.extras[self.config.qcelemental_full_qualifying_name_tag] = conformer_config.full_molecule_name
            qcmol.extras[self.config.qcelemental_current_stage_tag] = "02_generate-conformers"
            conformer_config.write_to_file(
                conformer_config.generated_conformer_qcschema_filename,
                contents=qcmol.json(),
            )

    def write_smiles(self):
        with open(self.filenames.smiles_filename, "w") as f:
            f.write(self.mapped_smiles)


@dataclasses.dataclass
class CanonicalSmilesMapper:
    """Class to map SMILES to canonical, numbered SMILES"""

    input_csv: str = str(DEFAULT_INPUT_CSV.resolve())
    output_mapped_smiles: str = str(DEFAULT_OUTPUT_MAPPINGS.resolve())

    def load_dataset(self):
        print(f"Loading dataset from {self.input_csv}")
        df = pd.read_csv(self.input_csv)
        return PhysicalPropertyDataSet.from_pandas(df)

    def generate_mapped_smiles(
        self,
        dataset: PhysicalPropertyDataSet
    ):
        all_components: Set[str] = {
            component.smiles
            for property in dataset.properties
            for component in property.substance.components
        }
        print(f"Found {len(all_components)} unique components")

        mapped_smiles = {}
        sorted_components = sorted(
            all_components,
            key=lambda x: (len(x), all_components),
        )
        for smiles in sorted_components:
            molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
            mapped = molecule.to_smiles(mapped=True)
            mapped_smiles[smiles] = mapped

        with open(self.output_mapped_smiles, "w") as f:
            json.dump(mapped_smiles, f, indent=4)
        print(f"Wrote to {self.output_mapped_smiles}")

        return mapped_smiles

    def run(self):
        dataset = self.load_dataset()
        self.generate_mapped_smiles(dataset)
        return self.output_mapped_smiles


@dataclasses.dataclass
class CanonicalGenerator:
    """Class to generate canonical conformers for molecules in a chunk"""

    input_mapped_smiles: str = str(DEFAULT_OUTPUT_MAPPINGS.resolve())
    output_csv: str = str(DEFAULT_OUTPUT_CSV.resolve())
    output_directory: str = DIRECTORY
    n_max_conformers: int = 10
    n_conformer_pool: int = 4000
    elf_percentage: float = 5.0
    conformer_rmsd_threshold: float = 0.05
    config: Config = Config()
    start_index: int = 0
    stop_index: Optional[int] = None

    def run(self):
        dataset = self.load_dataset()
        mapped_smiles = self.generate_mapped_smiles(dataset)
        self.generate_conformers(mapped_smiles)

    def load_mapped_smiles(self):
        with open(self.input_mapped_smiles, "r") as f:
            return json.load(f)

    def generate_conformers(self):
        all_mapped_smiles = self.load_mapped_smiles()
        all_keys = list(all_mapped_smiles.keys())
        n_keys = len(all_keys)
        stop_index = (
            n_keys if self.stop_index is None
            else max(self.stop_index, n_keys)
        )

        data = {
            "number": [],
            "smiles": [],
            "mapped_smiles": [],
            "component_name": [],
            "n_conformers": [],
        }

        for i in tqdm.tqdm(
            range(self.start_index, stop_index),
            total=stop_index - self.start_index,
            desc="Generating conformers",
        ):
            smiles = all_keys[i]
            mapped = all_mapped_smiles[smiles]

            number = i + 1
            conf_generator = ConformerGenerator(
                index=number,
                mapped_smiles=mapped,
                config=self.config,
                output_directory=self.output_directory,
                n_max_conformers=self.n_max_conformers,
                n_conformer_pool=self.n_conformer_pool,
                elf_percentage=self.elf_percentage,
                conformer_rmsd_threshold=self.conformer_rmsd_threshold,
            )
            n_confs = conf_generator.run()

            data["number"].append(number)
            data["smiles"].append(smiles)
            data["mapped_smiles"].append(mapped)
            data["component_name"].append(conf_generator.component_name)
            data["n_conformers"].append(n_confs)

        csvfile = self.output_csv
        if self.stop_index is not None:
            if csvfile.endswith("csv"):
                csvfile = csvfile[:-3]
            csvfile += f"{self.start_index + 1:03d}-{stop_index:03d}.csv"

        df = pd.DataFrame(data)
        df.to_csv(csvfile, index=False)
        print(f"Wrote to {csvfile}")
        return csvfile

    def run(self):
        return self.generate_conformers()


@click.group()
def cli():
    pass


@cli.command(
    help="Map SMILES to canonical, numbered SMILES"
)
@click.option(
    "--input-csv",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default=DEFAULT_INPUT_CSV,
    help="FreeSolv dataset CSV file, e.g. `subset_freesolv_output.csv`"
)
@click.option(
    "--output-mapped-smiles",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    default=DEFAULT_OUTPUT_MAPPINGS,
    help="Output JSON file with mapped SMILES, e.g. `mapped_smiles.json`"
)
def map_smiles(input_csv, output_mapped_smiles):
    mapper = CanonicalSmilesMapper(
        input_csv=input_csv,
        output_mapped_smiles=output_mapped_smiles,
    )
    output_file = mapper.run()
    print(output_file)


@cli.command(
    help="Generate canonical conformers for molecules in a chunk"
)
@click.option(
    "--input-mapped-smiles",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default=DEFAULT_OUTPUT_MAPPINGS,
    help="Output JSON file with mapped SMILES, e.g. `mapped_smiles.json`"
)
@click.option(
    "--output-csv",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    default=DEFAULT_OUTPUT_CSV,
    help="Output CSV file with conformers, e.g. `component_mappings.csv`"
)
@click.option(
    "--output-conformer-directory",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    default=DIRECTORY,
    help="Output directory for conformers, e.g. `.`"
)
@click.option(
    "--n-max-conformers",
    type=int,
    default=10,
    help="Max output conformers to generate"
)
@click.option(
    "--n-conformer-pool",
    type=int,
    default=4000,
    help="Max conformers to generate before filtering by ELF energy"
)
@click.option(
    "--elf-percentage",
    type=float,
    default=5.0,
    help="Percentage of conformers to keep based on ELF energy"
)
@click.option(
    "--conformer-rmsd-threshold",
    type=float,
    default=0.05,
    help="RMSD threshold for filtering conformers (Angstrom)"
)
@click.option(
    "--start-index",
    type=int,
    default=0,
    help="Start index for generating conformers (inclusive) -- use this if batching on supercomputer"
)
@click.option(
    "--stop-index",
    type=int,
    default=None,
    help="Stop index for generating conformers (exclusive) -- use this if batching on supercomputer"
)
def generate_conformers(
    input_mapped_smiles: str,
    output_csv: str,
    output_conformer_directory: str = DIRECTORY,
    n_max_conformers: int = 10,
    n_conformer_pool: int = 4000,
    elf_percentage: float = 5.0,
    conformer_rmsd_threshold: float = 0.05,
    start_index: int = 0,
    stop_index: Optional[int] = None,
):
    input_mapped_smiles = os.path.abspath(input_mapped_smiles)
    output_csv = os.path.abspath(output_csv)
    output_conformer_directory = os.path.abspath(output_conformer_directory)

    generator = CanonicalGenerator(
        input_mapped_smiles=input_mapped_smiles,
        output_csv=output_csv,
        output_directory=output_conformer_directory,
        n_max_conformers=n_max_conformers,
        n_conformer_pool=n_conformer_pool,
        elf_percentage=elf_percentage,
        conformer_rmsd_threshold=conformer_rmsd_threshold,
        start_index=start_index,
        stop_index=stop_index,
    )
    generator.run()


if __name__ == "__main__":
    cli()
