import dataclasses
import os
import pathlib
from typing import Union, Literal
import logging

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class Filenames:

    component_number: Union[int, Literal["*"]]
    conformer_number: Union[int, Literal["*"]] = 0
    charge_method_name: str = ""
    environment: str = ""

    component_name: str = "component-{component_number:03d}"
    conformer_name: str = "conformer-{conformer_number:02d}"
    full_molecule_name: str = "{component_name}_{conformer_name}"

    # conformer generation
    root_conformer_directory: str = "."
    conformer_directory: str = "{root_conformer_directory}/conformers/{component_name}"
    smiles_filename: str = "{conformer_directory}/{component_name}_smiles.smi"
    generated_conformer_qcschema_filename: str = "{conformer_directory}/{full_molecule_name}.qcschema"
    generated_conformer_xyz_filename: str = "{conformer_directory}/{full_molecule_name}.xyz"

    # # # === qm === # # #
    root_qm_directory: str = "."
    charge_environment_name: str = "{charge_method_name}-{environment}"
    qm_directory: str = "{root_qm_directory}/qm/{charge_environment_name}/{full_molecule_name}"

    # optimization
    optimized_conformer_qcschema_filename: str = "{qm_directory}/{full_molecule_name}_optimized.qcschema"
    optimized_conformer_xyz_filename: str = "{qm_directory}/{full_molecule_name}_optimized.xyz"
    optimization_options_json_filename: str = "{qm_directory}/optimization-options.json"

    # esp
    esp_grid_options_json_filename: str = "{qm_directory}/esp-grid-options.json"
    esp_grid_filename: str = "{qm_directory}/grid.dat"
    esp_esp_filename: str = "{qm_directory}/esp.dat"
    esp_orientation_filename: str = "{qm_directory}/{full_molecule_name}_orientation-esp.json"
    esp_options_json_filename: str = "{qm_directory}/esp-options.json"

    # # # === charges === # # # 
    root_charge_directory: str = "."
    component_charge_directory: str = "{root_charge_directory}/conformer_charges/{component_name}"
    conformer_charge_directory: str = "{component_charge_directory}/{full_molecule_name}"

    # resp
    resp_directory: str = "{component_charge_directory}/{charge_environment_name}"
    resp_conformer_directory: str = "{conformer_charge_directory}/{charge_environment_name}"
    surface_constraint_matrix_filename: str = "{resp_directory}/surface-constraint-matrix.dat"
    surface_constraint_matrix_conformer_filename: str = "{resp_conformer_directory}/surface-constraint-matrix.dat"
    atom_symmetries_filename: str = "{resp_directory}/atom-symmetries.json"

    stage_1_filename: str = "{resp_directory}/stage-1-respcharges.json"
    stage_1_conformer_filename: str = "{resp_conformer_directory}/stage-1-respcharges.json"
    stage_2_filename: str = "{resp_directory}/stage-2-respcharges.json"
    stage_2_conformer_filename: str = "{resp_conformer_directory}/stage-2-respcharges.json"
    resp_charges_filename: str = "{conformer_charge_directory}/{charge_environment_name}_charges.dat"
    resp_charges_short_filename: str = "{conformer_charge_directory}/{charge_environment_name}_short_charges.dat"

    # am1bcc
    # graph_directory: str = "{root_charge_directory}/graph"
    # am1bcc_charges_filename: str = "{graph_directory}/{full_molecule_name}_am1bcc-charges.dat"
    am1bcc_charges_filename: str = "{conformer_charge_directory}/am1bcc_charges.dat"

    charge_directory: str = "{root_charge_directory}/collated"
    collated_charge_filename: str = "{charge_directory}/collated_charges.csv"

    # atb
    atb_directory: str = "{conformer_charge_directory}/atb"
    atb_molid_file: str = "{atb_directory}/molid.dat"
    atb_lowest_molid_file: str = "{atb_directory}/lowest_molid.dat"
    atb_original_pdb_file: str = "{atb_directory}/{full_molecule_name}_original.pdb"
    atb_optimized_pdb_file: str = "{atb_directory}/{full_molecule_name}_optimized.pdb"
    atb_lowest_pdb_file: str = "{atb_directory}/{full_molecule_name}_lowest-optimized.pdb"
    atb_lowest_itp_file: str = "{atb_directory}/{full_molecule_name}_lowest.itp"
    atb_itp_file: str = "{atb_directory}/{full_molecule_name}.itp"
    atb_charges_filename: str = "{conformer_charge_directory}/atb_charges.dat"
    atb_lowest_charges_filename: str = "{conformer_charge_directory}/atb_lowest_charges.dat"
    atb_itp_smiles_filename: str = "{atb_directory}/conformer_itp_smiles.smi"
    atb_lowest_itp_smiles_filename: str = "{atb_directory}/lowest_itp_smiles.smi"


    # properties
    root_properties_directory: str = "."
    component_properties_directory: str = "{root_properties_directory}/properties/{component_name}"
    conformer_properties_directory: str = "{component_properties_directory}/{full_molecule_name}"
    property_esp_filename: str = "{conformer_properties_directory}/esp_{charge_method_name}_{conformer_name}.dat"
    property_dipole_filename: str = "{conformer_properties_directory}/dipole_{charge_method_name}_{conformer_name}.dat"


    # hfe files
    root_hfe_directory: str = "."
    forcefield: str = "atb"
    hfe_directory: str = "{root_hfe_directory}/hfe/{forcefield}/{full_molecule_name}"
    hfe_charge_directory: str = "{hfe_directory}/{charge_method_name}"
    hfe_working_directory: str = "{hfe_charge_directory}/03_hfe"
    hfe_u_nk_file: str = "{hfe_working_directory}/u_nk.csv"
    hfe_delta_f_file: str = "{hfe_working_directory}/mbar_delta_f.csv"
    hfe_d_delta_f_file: str = "{hfe_working_directory}/mbar_d_delta_f.csv"
    hfe_mbar_file: str = "{hfe_working_directory}/mbar.txt"

    hfe_xvg_pattern: str = "{hfe_charge_directory}/02_lambda*/09*.xvg"
    collated_hfe_file: str = "{root_hfe_directory}/collated_hfes.csv"

    # comparisons
    root_comparisons_directory: str = "."
    component_comparisons_directory: str = "{root_comparisons_directory}/comparisons/{component_name}"
    conformer_comparisons_directory: str = "{component_comparisons_directory}/{full_molecule_name}"

    collated_comparisons_file: str = "{root_comparisons_directory}/collated_comparisons.csv"

    def __post_init__(self):
        klass = type(self)
        vardict = dataclasses.asdict(self)
        for attrname, attrtype in klass.__annotations__.items():
            if attrtype is str:
                pattern = getattr(self, attrname)
                base_file = pattern.format(**vardict)
                vardict[attrname] = base_file
        
        for attrname, attr in vardict.items():
            if attrname.endswith("filename") and "." in attr:
                if os.path.abspath(attr) == attr:
                    pass
                attr = os.path.abspath(attr)
            setattr(self, attrname, attr)
            logger.debug(f"Setting {attrname} to {attr}")

    @staticmethod
    def make_parent(filename):
        """Make directory of file"""
        pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)

    def write_to_file(self, filename: str, contents: str):
        self.make_parent(filename)

        with open(filename, "w") as f:
            f.write(contents)
        print(f"Wrote to {filename}")
    