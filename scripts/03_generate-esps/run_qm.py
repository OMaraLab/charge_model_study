#!/usr/bin/env python3

import click
import dataclasses
import os
import numpy as np
import pathlib
import psiresp
from typing import Literal

import qcelemental as qcel

from charge_model_study.config import Config
from charge_model_study.qm import ChargeMethod
from charge_model_study.filenames import Filenames

DIRECTORY = pathlib.Path(__file__).parent.resolve()


@dataclasses.dataclass
class ESPFileGenerator:
    input_qcschema_file: str
    component_number: int
    conformer_number: int
    charge_method_name: str = "resp"
    environment: Literal["vacuum", "solvated"] = "vacuum"
    max_iter: int = 10000
    options_class = psiresp.qm.QMEnergyOptions
    config: Config = Config()
    output_directory: str = DIRECTORY

    def __post_init__(self):
        self.charge_method = ChargeMethod(self.charge_method_name.upper()).value
        self.qcmol = self.load_qcmol()
        if self.environment == "solvated":
            self.pcm_options = psiresp.PCMOptions()
        else:
            self.pcm_options = None

        self.filenames = Filenames(
            component_number=self.component_number,
            conformer_number=self.conformer_number,
            charge_method_name=self.charge_method_name,
            environment=self.environment,
            root_qm_directory=self.output_directory,
        )

    def run(self):
        jobdir = self.filenames.qm_directory
        pathlib.Path(jobdir).mkdir(parents=True, exist_ok=True)
        return self.write_job_files()

    @property
    def json_file(self):
        filename = f"{self.job_name}_{self.step}-options.json"
        return os.path.join(self.filenames.qm_directory, filename)

    def get_job_option_kwargs(self):
        return dict(
            jobname=self.job_name,
            pcm_options=self.pcm_options,
            method=self.charge_method.method,
            basis=self.charge_method.basis,
            keywords=dict(maxiter=self.max_iter)
        )

    def get_job_options(self):
        return self.options_class(**self.get_job_option_kwargs())

    @property
    def options_filename(self):
        return self.filenames.esp_options_json_filename

    def write_job_options(self):
        options = self.get_job_options()
        self.filenames.write_to_file(
            self.options_filename,
            options.json()
        )
        return options

    def write_job_files(self):
        options = self.write_job_options()
        filename = options.write_input(
            self.qcmol,
            working_directory=self.filenames.qm_directory,
        )
        print(f"Wrote to {filename}")
        return filename

    def load_qcmol(self):
        return qcel.models.Molecule.parse_file(self.input_qcschema_file)

    def postprocess(self):
        orientation = self.get_orientation_with_wavefunction()
        grid_options = self.get_grid_options()
        self.filenames.write_to_file(
            self.filenames.esp_grid_options_json_filename,
            grid_options.json()
        )

        orientation.compute_grid(grid_options=grid_options)
        self.filenames.make_parent(self.filenames.esp_grid_filename)
        np.savetxt(self.filenames.esp_grid_filename, orientation.grid)
        print(f"Wrote grid to {self.filenames.esp_grid_filename}")

        orientation.compute_esp()
        self.filenames.make_parent(self.filenames.esp_esp_filename)
        np.savetxt(self.filenames.esp_esp_filename, orientation.esp)
        print(f"Wrote ESP to {self.filenames.esp_esp_filename}")

        orientation.qcmol[self.config.qcelemental_current_stage_tag] = "03b-esp"
        self.filenames.write_to_file(
            self.filenames.esp_orientation_qcschema_filename,
            orientation.json()
        )
        return self.filenames.esp_orientation_qcschema_filename

    def get_grid_options(self):
        charge_method = self.charge_method
        return psiresp.GridOptions(
            use_radii=charge_method.vdw_radii,
            vdw_point_density=charge_method.vdw_point_density,
        )

    def get_orientation_with_wavefunction(self) -> psiresp.Orientation:
        options = self.get_job_options()
        wavefunction = options.run(
            qcmols=[self.qcmol],
            working_directory=self.filenames.qm_directory
        )[0]

        orientation = psiresp.Orientation(
            qcmol=self.qcmol,
            qc_wavefunction=wavefunction
        )
        return orientation


@dataclasses.dataclass
class OptimizationFileGenerator(ESPFileGenerator):
    g_convergence: str = "qchem"
    options_class = psiresp.qm.QMGeometryOptimizationOptions

    @property
    def options_filename(self):
        return self.filenames.optimization_options_json_filename

    def get_job_option_kwargs(self):
        options = super().get_job_option_kwargs()
        options["max_iter"] = self.max_iter
        options["g_convergence"] = self.g_convergence
        return options

    def postprocess(self):
        options = self.get_job_options()
        geometry = options.run(
            qcmols=[self.qcmol],
            working_directory=self.filenames.qm_directory,
        )[0]
        new_qcmol = self.qcmol.copy(update={"geometry": geometry})
        new_qcmol.extras[self.config.qcelemental_current_stage_tag] = "03a-optimized"

        self.filenames.write_to_file(
            self.filenames.optimized_conformer_qcschema_filename,
            new_qcmol.json()
        )
        return self.filenames.optimized_conformer_qcschema_filename


@click.group
@click.option(
    "--input-qcschema-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
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
    "--charge-method-name",
    type=click.Choice(["resp", "resp2"]),
    required=True,
)
@click.option(
    "--environment",
    type=click.Choice(["vacuum", "solvated"]),
    required=True,
)
@click.option(
    "--output-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=DIRECTORY,
)
@click.pass_context
def cli(
    ctx,
    input_qcschema_file,
    component_number,
    conformer_number,
    charge_method_name,
    solvated,
    output_directory,
):
    ctx.ensure_object(dict)
    ctx.obj["input_qcschema_file"] = input_qcschema_file
    ctx.obj["component_number"] = component_number
    ctx.obj["conformer_number"] = conformer_number
    ctx.obj["charge_method_name"] = charge_method_name
    ctx.obj["solvated"] = solvated
    ctx.obj["output_directory"] = output_directory


@cli.command()
@click.pass_context
def setup_geometry_optimization(ctx):
    qm_handler = OptimizationFileGenerator(**ctx.obj)
    filename = qm_handler.run()
    print(filename)


@cli.command()
@click.pass_context
def setup_wavefunction_computation(ctx):
    qm_handler = OptimizationFileGenerator(**ctx.obj)
    filename = qm_handler.postprocess()

    new_context = dict(**ctx.obj)
    new_context["input_qcschema_file"] = filename
    esp_handler = ESPFileGenerator(**new_context)
    psi4_filename = esp_handler.run()
    print(psi4_filename)


@cli.command()
@click.pass_context
def compute_esp(ctx):
    qm_handler = ESPFileGenerator(**ctx.obj)
    filename = qm_handler.postprocess()
    print(filename)


if __name__ == "__main__":
    cli(obj={})
