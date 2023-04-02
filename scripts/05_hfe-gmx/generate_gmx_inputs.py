import click
import pathlib


def generate_template_directory(
    output_directory: str,
    template_directory: str = "templates/preparation",
    force: bool = False,
    **kwargs
):
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    env = Environment(
        loader=FileSystemLoader(template_directory),
        autoescape=select_autoescape(),
        keep_trailing_newline=True,
    )
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)
    for template_name in env.list_templates():
        template = env.get_template(template_name)
        output_file = str(output_directory / template_name)
        output_template = env.from_string(output_file)
        output_file = pathlib.Path(output_template.render(**kwargs))
        output_file.parent.mkdir(exist_ok=True, parents=True)

        if output_file.exists() and not force:
            print(f"Skipping {output_file} because it already exists")
            continue

        with output_file.open("w") as f:
            f.write(template.render(**kwargs))
        print(f"Wrote {output_file}")


@click.group()
def cli():
    pass


@cli.command("generate-mdp-files")
@click.option(
    "--n-lambda",
    type=click.INT,
    default=20,
    show_default=True,
)
@click.option(
    "--force",
    is_flag=True,
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--template-mdp-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="templates/mdps",
    show_default=True,
)
@click.option(
    "--output-directory",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    default=".",
    show_default=True,
)
def generate_mdp_files(
    n_lambda=20,
    force=False,
    template_mdp_directory="templates/mdps",
    output_directory=".",
):
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    for lambda_value in range(n_lambda):
        generate_template_directory(
            output_directory=output_directory,
            template_directory=template_mdp_directory,
            lambda_value=lambda_value,
            force=force
        )


@cli.command("generate-molecule-files")
@click.option(
    "--molecule-name",
    type=str,
    required=True,
)
@click.option(
    "--charge-model",
    type=str,
    required=True,
)
@click.option(
    "--root-output-directory",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    default=".",
    show_default=True,
)
@click.option(
    "--template-preparation-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="templates/preparation",
    show_default=True,
)
@click.option(
    "--template-lambda-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="templates/lambda",
    show_default=True,
)
@click.option(
    "--n-lambda",
    type=click.INT,
    default=20,
    show_default=True,
)
@click.option(
    "--force",
    is_flag=True,
    type=bool,
    default=False,
    show_default=True,
)
def generate_input_files(
    molecule_name: str,
    charge_model: str,
    root_output_directory: str,
    template_preparation_directory: str = "templates/preparation",
    template_lambda_directory: str = "templates/lambda",
    n_lambda: int = 20,
    force: bool = False,
):
    root_output_directory = pathlib.Path(root_output_directory)
    root_output_directory.mkdir(exist_ok=True, parents=True)

    output_directory = root_output_directory / molecule_name / charge_model

    generate_template_directory(
        output_directory=output_directory / "01_preparation",
        template_directory=template_preparation_directory,
        molecule_name=molecule_name,
        charge_model=charge_model,
        force=force,
    )

    for lambda_value in range(n_lambda):
        generate_template_directory(
            output_directory=output_directory / f"02_lambda-{lambda_value:02d}",
            template_directory=template_lambda_directory,
            molecule_name=molecule_name,
            charge_model=charge_model,
            lambda_value=lambda_value,
            next_lambda_directory=f"02_lambda-{lambda_value + 1:02d}",
            force=force
        )


if __name__ == "__main__":
    cli()
