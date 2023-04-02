import dataclasses
import glob
import pathlib
import os
from typing import List

import click
import pandas as pd
import numpy as np
import tqdm

from openff.units import unit
from charge_model_study.filenames import Filenames

import warnings

# /Users/lily/anaconda3/envs/charge-model-study/lib/python3.9/site-packages/alchemlyb/parsing/gmx.py:103: FutureWarning: pandas.Float64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.
warnings.simplefilter(action='ignore', category=FutureWarning)


kb = 1.380649e-23 * unit.J / unit.K
kbt = kb * 298.15 * unit.K
rt = kbt * 6.02214076e23 / unit.mol
KB_T = rt.m_as(unit.kJ / unit.mol)  # 2.4789570296023884


def check_groups(smi, functional_groups=["Alcohol", "Amine", "Amide", "Carboxylic Acid", "Ether", "Ester", "Nitro Compound", "Nitrate", "Aromatic", "Heterocycle"]):
    from openff.evaluator.utils.checkmol import analyse_functional_groups
    groups = analyse_functional_groups(smi)
    groups = {x.value: v for x, v in groups.items()}
    outputs = {}
    for k in functional_groups:
        if k in groups:
            outputs[k] = True
        elif any(x.endswith(k) for x in groups):
            outputs[k] = True
        else:
            outputs[k] = False

    if "Phenol" in groups:
        outputs["Alcohol"] = True

    if "Alkylarylether" in groups:
        outputs["Ether"] = True
    return outputs


@dataclasses.dataclass
class HFECalculator:
    root_qm_directory: str
    root_hfe_directory: str
    root_properties_directory: str
    component_number: int
    conformer_number: int
    forcefield: str
    charge_method: str
    n_lambda: int = 20
    temperature: float = 298.15
    force: bool = False

    def __post_init__(self):
        self.filenames = Filenames(
            component_number=self.component_number,
            conformer_number=self.conformer_number,
            charge_method_name=self.charge_method,
            root_hfe_directory=self.root_hfe_directory,
            root_qm_directory=self.root_qm_directory,
            root_properties_directory=self.root_properties_directory,
            forcefield=self.forcefield,
        )
        pathlib.Path(self.filenames.hfe_working_directory).mkdir(
            parents=True, exist_ok=True)

    def run(self):
        import alchemlyb as alc
        from alchemlyb.estimators import AutoMBAR
        from alchemlyb.parsing.gmx import extract_u_nk

        if not self.force:
            mbar_file = self.filenames.hfe_mbar_file
            if os.path.exists(mbar_file):
                with open(mbar_file, "r") as f:
                    contents = f.read()
                try:
                    before, after = contents.split("±")
                    value = float(before.strip())
                    error = float(after.split("kJ")[0].strip())
                    return (value, error)
                except:
                    pass

        xvgs = sorted(glob.glob(str(self.filenames.hfe_xvg_pattern)))

        if len(xvgs) < self.n_lambda:
            raise ValueError(
                f"Could not compute HFE for {self.filenames.full_molecule_name} "
                f"with charge model {self.charge_method}. "
                f"Expected {self.n_lambda} lambda windows, found {len(xvgs)}: {xvgs}"
            )
        dfs = [extract_u_nk(x, self.temperature) for x in xvgs]
        df = alc.concat(dfs)
        # df.to_csv(str(self.filenames.hfe_u_nk_file))

        mbar = AutoMBAR().fit(df)
        # mbar.delta_f_.to_csv(str(self.filenames.hfe_delta_f_file))
        # mbar.d_delta_f_.to_csv(str(self.filenames.hfe_d_delta_f_file))

        value = -mbar.delta_f_.loc[[(0.0, 0.0)],
                                   [(1.0, 1.0)]].values[0][0] * KB_T
        error = mbar.d_delta_f_.loc[[(0.0, 0.0)], [
            (1.0, 1.0)]].values[0][0] * KB_T

        contents = f"{value} ± {error} kJ/mol"
        with open(self.filenames.hfe_mbar_file, "w") as f:
            f.write(contents)
        print(
            f"Computed HFE for {self.filenames.full_molecule_name} "
            f"with charge model {self.charge_method}: {contents}")
        return value, error

    def _load_properties(
        self,
        charge_method: str,
        property_name: str,
        conformer_number: int = 0
    ):
        if charge_method == "atb_conformer":
            conformer_number = 1
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

    def _load_dipoles(self):
        dipoles = self._load_properties(
            charge_method=self.charge_method,
            property_name="dipole",
            conformer_number=self.conformer_number
        )
        dipole_distances = [
            np.linalg.norm(dipole)
            for dipole in dipoles
        ]
        return dipole_distances

    def load_dipoles(self):
        dipole_distances = self._load_dipoles()
        mean_std = (np.mean(dipole_distances), np.std(dipole_distances))
        return mean_std

    def _load_esps(self):
        return self._load_properties(
            charge_method=self.charge_method,
            property_name="esp",
            conformer_number=self.conformer_number
        )

    def _load_qm_esps(self):
        from openff.units import unit

        pattern = os.path.join(
            self.filenames.root_qm_directory,
            "qm",
            "resp-vacuum",
            f"{self.filenames.component_name}_*",
            "esp.dat"
        )
        files = sorted(glob.glob(pattern))
        return [
            np.loadtxt(file)
            for file in files
        ]

    def _load_qm_dipoles(self):
        from openff.units import unit

        pattern = os.path.join(
            self.filenames.root_qm_directory,
            "qm",
            "resp-vacuum",
            f"{self.filenames.component_name}_*",
            "single_point",
            "dipole_au.dat"
        )
        files = sorted(glob.glob(pattern))

        psi4_unit = 8.478353552E-30 * unit.C * unit.m
        dipoles = [
            (np.loadtxt(file) * psi4_unit).m_as(unit.debye)
            for file in files
        ]
        dipole_distances = [
            np.linalg.norm(dipole)
            for dipole in dipoles
        ]
        return dipole_distances

    def load_qm_dipoles(self):
        dipole_distances = self._load_qm_dipoles()
        mean_std = (np.mean(dipole_distances), np.std(dipole_distances))
        return mean_std

    def load_qm_dipole_differences(self):
        dipoles = self._load_dipoles()
        qm_dipoles = self._load_qm_dipoles()
        differences = np.array(dipoles) - np.array(qm_dipoles)
        mean_std = (np.mean(differences), np.std(differences))
        return mean_std

    def get_esp_differences(self):
        esps = np.concatenate(self._load_esps())
        qm_esps = np.concatenate(self._load_qm_esps())

        diff = esps - qm_esps
        return (diff ** 2).mean() ** 0.5


def lookup_experimental_values(
    freesolv_df,
    component_mappings,
    component_number: int
):
    all_keys = list(component_mappings.keys())
    smiles = all_keys[component_number - 1]
    mapped_smiles = component_mappings[smiles]

    row = freesolv_df[freesolv_df["Component 2"] == smiles]
    value = row["SolvationFreeEnergy Value (kJ / mol)"].values[0]
    error = row["SolvationFreeEnergy Uncertainty (kJ / mol)"].values[0]
    source = row["Source"].values[0]
    return {
        "experimental_value": value,
        "experimental_error": error,
        "source": source,
        "smiles": smiles,
        "mapped_smiles": mapped_smiles
    }


@click.command()
@click.option(
    "--root-hfe-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing the HFE files. See charge_model_study.filenames.Filenames",
)
@click.option(
    "--root-qm-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing the QM files. See charge_model_study.filenames.Filenames",
)
@click.option(
    "--root-properties-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory for previously computed properties. See charge_model_study.filenames.Filenames",
)
@click.option(
    "--forcefield",
    "forcefields",
    multiple=True,
    type=str,
)
@click.option(
    "--component-mapping-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Component mapping JSON file, e.g. `mapped_smiles.json`",
)
@click.option(
    "--freesolv-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Freesolv CSV file, e.g. `subset_freesolv_output.csv`",
)
@click.option(
    "--force",
    help="recompute even if previously computed hfe file is found",
    is_flag=True,
    type=bool,
    default=False,
    show_default=True,
)
def compute_all_mbar(
    root_hfe_directory: str,
    root_qm_directory: str,
    root_properties_directory: str,
    forcefields: List[str],
    component_mapping_file: str,
    freesolv_file: str,
    force=False
):
    import json

    dataset = pd.read_csv(freesolv_file)
    with open(component_mapping_file, "r") as f:
        component_mappings = json.load(f)

    base_fn = Filenames(
        component_number=0,
        conformer_number=0,
        root_hfe_directory=root_hfe_directory,
    )
    results = []

    for ff in forcefields:
        fn = Filenames(
            component_number=0,
            conformer_number=0,
            full_molecule_name="*",
            charge_method_name="*",
            root_hfe_directory=root_hfe_directory,
            forcefield=ff,
        )
        print(fn.hfe_xvg_pattern)
        xvgs = [pathlib.Path(path)
                for path in glob.glob(str(fn.hfe_xvg_pattern))]
        component_charges = sorted(
            set(
                tuple(str(path.parent.parent).split("/")[-2:])
                for path in xvgs
            )
        )

        n_incomplete = 0
        for component_name, charge_model in tqdm.tqdm(
            component_charges,
            desc=f"computing {ff} HFEs"
        ):
            component, conformer = component_name.split("_")
            component_number = int(component.split("-")[-1])
            conformer_number = int(conformer.split("-")[-1])

            if component_number == 2:
                # this is water, oops
                continue

            calculator = HFECalculator(
                root_hfe_directory=root_hfe_directory,
                root_qm_directory=root_qm_directory,
                root_properties_directory=root_properties_directory,
                component_number=component_number,
                conformer_number=0,
                charge_method=charge_model,
                force=force,
                forcefield=ff
            )

            try:
                value, error = calculator.run()
            except ValueError as e:
                print(e)
                n_incomplete += 1
                continue

            dipole_mean, dipole_std = calculator.load_dipoles()
            qm_dipole_mean, qm_dipole_std = calculator.load_qm_dipoles()
            dipole_diff_mean, dipole_diff_std = calculator.load_qm_dipole_differences()

            row = {
                "component_name": component_name,
                "component_number": component_number,
                "conformer_number": conformer_number,
                "charge_model": charge_model,
                "value": value,
                "error": error,
                "forcefield": ff,
                "dipole_mean": dipole_mean,
                "dipole_std": dipole_std,
                "qm_dipole_mean": qm_dipole_mean,
                "qm_dipole_std": qm_dipole_std,
                "dipole_diff_mean": dipole_diff_mean,
                "dipole_diff_std": dipole_diff_std,
                "esp_rmse": calculator.get_esp_differences(),
            }
            experimental = lookup_experimental_values(
                dataset,
                component_mappings,
                component_number
            )
            row.update(experimental)
            groups = check_groups(experimental["mapped_smiles"])
            row.update(groups)
            results.append(row)

        print(f"Could not compute {n_incomplete} HFEs for {ff}.")

    df = pd.DataFrame(results)
    df.to_csv(str(base_fn.collated_hfe_file))

    print(f"Saved results to {str(base_fn.collated_hfe_file)}")


if __name__ == "__main__":
    compute_all_mbar()
