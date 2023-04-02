import click


def write_water_itp():
    from openff.toolkit.topology import Molecule
    from openff.toolkit.typing.engines.smirnoff import ForceField

    water = Molecule.from_smiles("O")
    forcefield = ForceField("openff-2.0.0.offxml")
    interchange = forcefield.create_interchange(water.to_topology())
    interchange.to_top("tip3p.itp")


def edit_openff_top(top_file):

    with open(top_file, "r") as f:
        content = f.read()

    before_moleculetype, after_moleculetype = content.split("[ moleculetype ]")
    include_path = [
        '',
        '#include "../../../../../openff-2.0.0.ff/tip3p.itp"',
        '',
        '[ moleculetype ]',
    ]
    new_content = (
        before_moleculetype
        + "\n".join(include_path)
        + after_moleculetype
    )
    with open(top_file, "w") as f:
        f.write(new_content)

    print(f"Edited {top_file} to include tip3p.itp")


@click.command()
@click.option(
    "--top-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
def cli(top_file):
    edit_openff_top(top_file)


if __name__ == "__main__":
    cli()
