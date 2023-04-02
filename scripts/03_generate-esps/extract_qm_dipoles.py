
import click
import glob
import pathlib


def extract_dipole(
    msgpack: str,
    output_filename: str = "dipole_au.dat",
):
    """
    Extract the dipole from a QCEngine msgpack file.
    """
    import qcelemental as qcel
    import numpy as np

    with open(msgpack, "rb") as f:
        content = f.read()

    data = qcel.util.deserialize(content, "msgpack")
    dipole = data["extras"]["qcvars"]["SCF DIPOLE"]

    output_directory = pathlib.Path(msgpack).parent
    output_file = str(output_directory / output_filename)
    np.savetxt(output_file, dipole)


@click.command()
@click.option(
    "--qm-directory",
    type=str,
    required=True,
    help="Directory containing QM calculations -- see charge_model_study.filenames for more",
)
def extract_all_dipoles(
    qm_directory: str,
):
    import glob
    import tqdm
    import pathlib

    qm_directory = pathlib.Path(qm_directory)
    msgpacks = glob.glob(str(qm_directory / "**" / "*.msgpack"), recursive=True)
    for msgpack in tqdm.tqdm(msgpacks):
        extract_dipole(msgpack)


if __name__ == "__main__":
    extract_all_dipoles()
