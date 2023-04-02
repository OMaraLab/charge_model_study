import dataclasses
import yaml


@dataclasses.dataclass
class Config:
    qcelemental_component_tag: str = "component_number"
    qcelemental_conformer_tag: str = "conformer_number"
    qcelemental_full_qualifying_name_tag: str = "full_conformer_name"
    qcelemental_current_stage_tag: str = "stage"
    qcelemental_conformer_energy_tag: str = "mmff94_electrostatic_energy"
    qcelemental_mapped_smiles_tag: str = "canonical_isomeric_explicit_hydrogen_mapped_smiles"

    @classmethod
    def from_file(cls, file):
        with open(file, "r") as f:
            contents = yaml.safe_load(f)
        contents = {
            k.replace("-", "_"): v
            for k, v in contents.items()
        }
        return cls(**contents)
