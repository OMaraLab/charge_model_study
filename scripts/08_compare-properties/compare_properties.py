import dataclasses
import os
import pathlib
import itertools
import glob

import tqdm
import pandas as pd
import click
import numpy as np

from openff.toolkit import Molecule
from charge_model_study.filenames import Filenames
from openff.toolkit import ForceField
from openff.units import unit


@dataclasses.dataclass
class Comparisons:
    component_number: int
    root_properties_directory: str
    root_conformer_directory: str
    root_hfe_directory: str
    root_charge_directory: str

    def __post_init__(self):
        self.filenames = Filenames(
            component_number=self.component_number,
            conformer_number=1,
            root_conformer_directory=self.root_conformer_directory,
            root_properties_directory=self.root_properties_directory,
            root_hfe_directory=self.root_hfe_directory,
            root_charge_directory=self.root_charge_directory
        )

        with open(self.filenames.smiles_filename, "r") as f:
            self.smiles = f.read().strip()
        self.offmol = Molecule.from_mapped_smiles(
            self.smiles, allow_undefined_stereo=True)
        df = pd.read_csv(self.filenames.collated_charge_filename)
        self.charges = df[
            (df.component_number == self.component_number)
            & (df.conformer_number == 0)
        ]
        hfes = pd.read_csv(self.filenames.collated_hfe_file)
        self.hfes = hfes[hfes.component_number == self.component_number]

        ff = ForceField("openff-2.0.0.offxml")
        top = ff.label_molecules(self.offmol.to_topology())[0]
        n_atoms = len(self.offmol.atoms)
        openff_epsilon = [None] * n_atoms
        for k, v in top["vdW"].items():
            openff_epsilon[k[0]] = v.epsilon.m_as(unit.kilocalorie_per_mole)
        self.openff_epsilon = np.array(openff_epsilon)

    def _load_properties(
        self,
        charge_method: str,
        property_name: str,
        conformer_number: int = 0
    ):
        pattern = os.path.join(
            self.filenames.component_properties_directory,
            "*",
            f"{property_name}_{charge_method}_conformer-{conformer_number:02d}.dat"
        )
        files = sorted(glob.glob(pattern))
        properties = [
            np.loadtxt(file)
            for file in files
        ]
        return properties

    def load_charges(self, charge_method):
        charges = self.charges[self.charges.charge_method == charge_method]
        return charges.sort_values("atom_index").charge.values

    def load_esps(self, charge_method, conformer_number: int = 0):
        return self._load_properties(
            charge_method=charge_method,
            property_name="esp",
            conformer_number=conformer_number,
        )

    def load_dipoles(self, charge_method, conformer_number: int = 0):
        return self._load_properties(
            charge_method=charge_method,
            property_name="dipole",
            conformer_number=conformer_number,
        )

    def compare_esps(self, charge_method_1, charge_method_2):
        esps_1 = self.load_esps(charge_method_1)
        esps_2 = self.load_esps(charge_method_2)
        assert len(esps_1) == len(esps_2)

        rmses = []
        for x, y in zip(esps_1, esps_2):
            diff = x - y
            rmse = (diff ** 2).mean() ** 0.5
            rmses.append(rmse)
        return np.mean(rmses)

    def compare_concat_esps(self, charge_method_1, charge_method_2):
        esps_1 = np.concatenate(self.load_esps(charge_method_1))
        esps_2 = np.concatenate(self.load_esps(charge_method_2))
        diff = esps_1 - esps_2
        return (diff ** 2).mean() ** 0.5

    def compare_norm_esps(self, charge_method_1, charge_method_2):
        esps_1 = np.concatenate(self.load_esps(charge_method_1))
        esps_2 = np.concatenate(self.load_esps(charge_method_2))
        denom = (np.abs(esps_1) + np.abs(esps_2)) / 2
        diff = (esps_1 - esps_2) / denom
        return (diff ** 2).mean() ** 0.5

    def compare_dipole_angle(self, charge_method_1, charge_method_2):
        dipoles_1 = self.load_dipoles(charge_method_1)
        dipoles_2 = self.load_dipoles(charge_method_2)
        assert len(dipoles_1) == len(dipoles_2)

        angles = []
        for x, y in zip(dipoles_1, dipoles_2):
            cosine = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
            cosine = np.clip(cosine, -1.0, 1.0)
            angle = np.arccos(cosine)
            angles.append(np.abs(angle))
        return np.mean(angles)

    def compare_charges(self, charges_1, charges_2):
        diff = charges_1 - charges_2
        return (diff ** 2).mean() ** 0.5

    def compare_charges_lj(self, charges_1, charges_2):
        diff = charges_1 - charges_2
        return ((diff ** 2) * self.openff_epsilon).mean() ** 0.5

    def compare_max_difference(self, charges_1, charges_2):
        diff = charges_1 - charges_2
        return np.max(np.abs(diff))

    def compare_max_heavy_difference(self, charges_1, charges_2):
        mask = [atom.symbol != "H" for atom in self.offmol.atoms]
        diff = (charges_1 - charges_2)[mask]
        return np.max(np.abs(diff))

    def compare_non_hydrocarbon_max_difference(self, charges_1, charges_2):
        symbols = [atom.symbol for atom in self.offmol.atoms]
        mask = [symbol not in ["H", "C"] for symbol in symbols]
        diff = (charges_1 - charges_2)[mask]
        return np.max(np.abs(diff))

    def compare_heavy_charges(self, charges_1, charges_2):
        mask = [atom.symbol != "H" for atom in self.offmol.atoms]
        diff = (charges_1 - charges_2)[mask]
        mse = np.mean(diff * diff)
        return np.sqrt(mse)

    def compare_non_hydrocarbon_rmse(self, charges_1, charges_2):
        symbols = [atom.symbol for atom in self.offmol.atoms]
        mask = [symbol not in ["H", "C"] for symbol in symbols]
        diff = (charges_1 - charges_2)[mask]
        mse = np.mean(diff * diff)
        return np.sqrt(mse)

    def compare_norm_rmse(self, charges_1, charges_2):
        diff = (charges_1 - charges_2)
        denom = (charges_1 + charges_2)/2
        norm = (diff * denom)
        mse = np.mean(norm * norm)
        return np.sqrt(mse)

    def compare_norm_rmse_std(self, charges_1, charges_2):
        diff = (charges_1 - charges_2)
        denom = (charges_1 + charges_2)/2
        norm = (diff * denom * np.std(denom))
        mse = np.mean(norm * norm)
        return np.sqrt(mse)

    def compare_norm_non_hydrocarbon_rmse(self, charges_1, charges_2):
        symbols = [atom.symbol for atom in self.offmol.atoms]
        mask = [symbol not in ["H", "C"] for symbol in symbols]
        diff = (charges_1 - charges_2)
        denom = (charges_1 + charges_2)/2
        norm = (diff * denom * np.std(denom))[mask]
        mse = np.mean(norm * norm)
        return np.sqrt(mse)

    def compare_norm_non_hydrocarbon_rmse_abs(self, charges_1, charges_2):
        symbols = [atom.symbol for atom in self.offmol.atoms]
        mask = [symbol not in ["H", "C"] for symbol in symbols]
        diff = (charges_1 - charges_2)
        denom = (np.abs(charges_1) + np.abs(charges_2))/2
        norm = (diff * denom * np.std(denom))[mask]
        mse = np.mean(norm * norm)
        return np.sqrt(mse)

    def run(self, charge_method_1, charge_method_2):
        charges_1 = self.load_charges(charge_method_1)
        charges_2 = self.load_charges(charge_method_2)

        hfe_1_df = self.hfes[self.hfes.charge_model == charge_method_1]
        hfe_1_atb = hfe_1_df[hfe_1_df.forcefield == "atb"]
        assert len(hfe_1_atb) == 1
        hfe_1_atb = hfe_1_atb.value.values[0]

        hfe_2_df = self.hfes[self.hfes.charge_model == charge_method_2]
        hfe_2_atb = hfe_2_df[hfe_2_df.forcefield == "atb"]
        assert len(hfe_2_atb) == 1
        hfe_2_atb = hfe_2_atb.value.values[0]

        hfe_diff_atb = np.abs(hfe_1_atb - hfe_2_atb)

        try:
            hfe_1_openff = hfe_1_df[hfe_1_df.forcefield ==
                                    "openff-2.0.0"]
            assert len(hfe_1_openff) == 1
            hfe_1_openff = hfe_1_openff.value.values[0]

            hfe_2_openff = hfe_2_df[hfe_2_df.forcefield ==
                                    "openff-2.0.0"]
            assert len(hfe_2_openff) == 1
            hfe_2_openff = hfe_2_openff.value.values[0]
        except:
            hfe_diff_openff = np.nan
        else:
            hfe_diff_openff = np.abs(hfe_1_openff - hfe_2_openff)

        return {
            "component_number": self.component_number,
            "component_name": f"component-{self.component_number:03d}_conformer-00",
            "mapped_smiles": self.smiles,
            "experimental_value": self.hfes.experimental_value.values[0],
            "atb_diff": hfe_diff_atb,
            "openff_diff": hfe_diff_openff,
            "charge_method_1": charge_method_1,
            "charge_method_2": charge_method_2,
            "n_atoms": len(self.offmol.atoms),
            "n_heavy_atoms": len([atom for atom in self.offmol.atoms if atom.symbol != "H"]),
            "esp_rmse": self.compare_esps(charge_method_1, charge_method_2),
            "concat_esp": self.compare_concat_esps(charge_method_1, charge_method_2),
            "dipole_angle": self.compare_dipole_angle(charge_method_1, charge_method_2),
            "charge_rmse": self.compare_charges(charges_1, charges_2),
            "lj_charge_rmse": self.compare_charges_lj(charges_1, charges_2),
            "max_difference": self.compare_max_difference(charges_1, charges_2),
            "max_heavy_difference": self.compare_max_heavy_difference(charges_1, charges_2),
            "non_hydrocarbon_max_difference": self.compare_non_hydrocarbon_max_difference(charges_1, charges_2),
            "heavy_charge_rmse": self.compare_heavy_charges(charges_1, charges_2),
            "non_hydrocarbon_rmse": self.compare_non_hydrocarbon_rmse(charges_1, charges_2),
            "norm_non_hydrocarbon_rmse": self.compare_norm_non_hydrocarbon_rmse(charges_1, charges_2),
            "norm_rmse": self.compare_norm_rmse(charges_1, charges_2),
            "norm_rmse_std": self.compare_norm_rmse_std(charges_1, charges_2),
            "norm_non_hydrocarbon_rmse_abs": self.compare_norm_non_hydrocarbon_rmse_abs(charges_1, charges_2),
            "norm_esps": self.compare_norm_esps(charge_method_1, charge_method_2),
        }


@click.command()
@click.option(
    "--root-properties-directory",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="The root directory containing the properties files.",
)
@click.option(
    "--root-conformer-directory",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="The root directory containing the conformer files.",
)
@click.option(
    "--root-hfe-directory",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="The root directory containing the hfe files.",
)
@click.option(
    "--root-charge-directory",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="The root directory containing the charge files.",
)
@click.option(
    "--root-comparisons-directory",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="The root directory containing the comparisons files.",
)
def compare_all(
    root_properties_directory: str,
    root_conformer_directory: str,
    root_hfe_directory: str,
    root_charge_directory: str,
    root_comparisons_directory: str,
):
    fn = Filenames(
        component_number=0,
        conformer_number=0,
        root_hfe_directory=root_hfe_directory,
        root_comparisons_directory=root_comparisons_directory,
    )
    hfes = pd.read_csv(fn.collated_hfe_file)

    rows = []
    for component, subdf in tqdm.tqdm(hfes.groupby("component_number")):
        charge_methods = sorted(subdf.charge_model.unique())

        comparer = Comparisons(
            component_number=component,
            root_properties_directory=root_properties_directory,
            root_conformer_directory=root_conformer_directory,
            root_hfe_directory=root_hfe_directory,
            root_charge_directory=root_charge_directory,
        )
        for charge_method_1, charge_method_2 in itertools.combinations(charge_methods, 2):
            try:
                row = comparer.run(charge_method_1, charge_method_2)
            except AssertionError:
                continue
            else:
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(fn.collated_comparisons_file)


if __name__ == "__main__":
    compare_all()
