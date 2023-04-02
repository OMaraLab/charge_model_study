from collections import defaultdict
import dataclasses
from typing import Tuple
import json
import scipy
import numpy as np


@dataclasses.dataclass
class AtomSymmetries:
    methyl_carbon_indices: Tuple[int, ...]
    methyl_hydrogen_indices: Tuple[int, ...]
    heavy_hydrogen_indices: Tuple[int, ...]
    heavy_atom_indices: Tuple[int, ...]
    atom_symmetries: Tuple[int, ...]

    @property
    def heavy_indices(self):
        return self.heavy_atom_indices + self.heavy_hydrogen_indices

    @classmethod
    def from_openff_molecule(cls, offmol, atom_symmetries: Tuple[int, ...]):
        # methyl carbons
        pattern = "[#6X4H3,#6H4,#6X4H2:1]"
        methyl_carbon_indices = {
            x[0]
            for x in offmol.chemical_environment_matches(pattern)
        }

        # methyl hydrogens
        h_bonded_indices = {
            index: {neighbor.molecule_atom_index for neighbor in atom.bonded_atoms}
            for index, atom in enumerate(offmol.atoms)
            if atom.atomic_number == 1
        }
        methyl_hydrogen_indices = {
            hydrogen_index
            for hydrogen_index, neighbor_indices in h_bonded_indices.items()
            if methyl_carbon_indices & neighbor_indices
        }

        heavy_hydrogen_indices = set(h_bonded_indices) - methyl_hydrogen_indices
        heavy_atom_indices = (
            set(range(len(offmol.atoms)))
            - methyl_carbon_indices
            - methyl_hydrogen_indices
            - heavy_hydrogen_indices
        )

        return cls(
            methyl_carbon_indices=tuple(sorted(methyl_carbon_indices)),
            methyl_hydrogen_indices=tuple(sorted(methyl_hydrogen_indices)),
            heavy_hydrogen_indices=tuple(sorted(heavy_hydrogen_indices)),
            heavy_atom_indices=tuple(sorted(heavy_atom_indices)),
            atom_symmetries=atom_symmetries,
        )

    def json(self):
        return json.dumps({
            k: list(v)
            for k, v in dataclasses.asdict(self).items()
        })

    @classmethod
    def parse_file(cls, filename: str):
        with open(filename, "r") as f:
            contents = f.read()
        loaded = {
            k: tuple(v)
            for k, v in json.loads(contents).items()
        }
        return cls(**loaded)

    def generate_stage_1_symmetries(self, exclude_methyl_hydrogens: bool = True):
        """Exclude methyl hydrogens for 2-stage RESP; include them for 1-stage ESP."""
        # all atoms constrained to equivalent
        # except methyl Hs
        symmetries = list(self.atom_symmetries)
        if exclude_methyl_hydrogens:
            max_index = max(symmetries) + 1
            for index in self.methyl_hydrogen_indices:
                symmetries[index] = max_index
                max_index += 1
        return symmetries

    def generate_stage_1_constraints(self, exclude_methyl_hydrogens: bool = True):
        equivalences = self.generate_stage_1_symmetries(
            exclude_methyl_hydrogens)
        constraints = self.generate_equivalence_equations_from_symmetries(
            equivalences)

        targets = np.zeros(len(constraints))
        return self.add_total_charge_constraint(constraints, targets)

    @staticmethod
    def add_total_charge_constraint(constraints, targets):
        # ones = np.ones((1, constraints.shape[1]))
        # constraints = np.vstack([constraints, ones])
        # targets = np.concatenate([targets, [0]])
        return constraints, targets

    def generate_stage_2_symmetries(self):
        # methyl carbons and hydrogens allowed to vary
        # all other atoms are fixed to stage 1 charges
        symmetries = list(self.atom_symmetries)
        max_index = max(symmetries) + 1
        for index in self.heavy_indices:
            symmetries[index] = max_index
            max_index += 1
        return symmetries

    def generate_stage_2_constraints(self, charges):
        symm = self.generate_stage_2_symmetries()
        equivs = self.generate_equivalence_equations_from_symmetries(symm)

        heavy_indices = self.heavy_indices
        constrained_charges = []
        n_dim = len(self.atom_symmetries) + 1
        constraints = np.zeros((len(heavy_indices), n_dim))
        for i, index in enumerate(self.heavy_indices):
            constrained_charges.append(charges[index])
            constraints[i, index] = 1

        if len(equivs):
            all_constraints = np.vstack([constraints, equivs])
        else:
            all_constraints = constraints
        all_targets = np.zeros(len(all_constraints))
        all_targets[:len(constrained_charges)] = constrained_charges

        return self.add_total_charge_constraint(all_constraints, all_targets)

    def _empty_row(self):
        return np.zeros_like(self.atom_symmetries)

    @staticmethod
    def generate_equivalence_equations_from_symmetries(atom_symmetries) -> np.ndarray:
        indices = defaultdict(list)
        for i, group in enumerate(atom_symmetries):
            indices[group].append(i)

        equivalent = {k: v for k, v in indices.items() if len(v) > 1}
        constraints = []
        n_dim = len(atom_symmetries) + 1
        for indices in equivalent.values():
            empty = np.zeros((len(indices) - 1, n_dim))
            for i, (j, k) in enumerate(zip(indices[:-1], indices[1:])):
                empty[i][j] = 1
                empty[i][k] = -1
            constraints.append(empty)

        if constraints:
            return np.vstack(constraints)
        return np.array([])

    @staticmethod
    def combine_constraints(surface, constraints, targets):
        # combine the surface constraints and... constraint constraints
        # into one big sparse matrix

        constraint_matrix = scipy.sparse.csr_matrix(surface.coefficient_matrix)
        constraint_values = surface.constant_vector

        if sum(constraints.shape):
            charge_matrix = scipy.sparse.csr_matrix(constraints)
            constraint_matrix = scipy.sparse.bmat(
                [
                    [constraint_matrix, charge_matrix.transpose()],
                    [charge_matrix, None],
                ],
                format="csr"
            )

            constraint_values = np.concatenate([constraint_values, targets])

        assert len(constraint_values) == constraint_matrix.shape[1]
        return constraint_matrix, constraint_values

    def get_hydrogen_mask(self):
        mask = np.ones_like(self.atom_symmetries, dtype=bool)
        for index in self.heavy_hydrogen_indices + self.methyl_hydrogen_indices:
            mask[index] = False
        return mask
