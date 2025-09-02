"""!
@file
PyPO docs generator.

For this script to work properly, you should have installed the docs prerequisites.
"""

from pathlib import Path
import shutil
import subprocess
import traceback

# Paths/names of command-line tools
DOXYGEN = "doxygen"
JUPYTER = "jupyter"

def check_deps():
    for call in ((DOXYGEN, "--version"), (JUPYTER, "--version")):
        try:
            subprocess.run(
                call,
                shell=False,
                check=True,
                )
        except (subprocess.CalledProcessError, FileNotFoundError) as err:
            raise Exception(
                f"External dependency `{call[0]}` is missing or broken, "
                "go fix it!"
                ) from err




def generate_docs():
    base_path = Path(__file__).resolve().parent
    doc_path = base_path / "docs"
    demo_path = base_path / "etc" / "demos"
    doxy_path = base_path / "doxy"

    shutil.rmtree(doc_path)

    ### Convert jupyter to html ###

    for file in demo_path.iterdir():
        if not (file.is_file() and file.suffix == ".ipynb"):
            continue

        dest_path = doc_path / "demos"
        dest_path.mkdir(parents=True, exist_ok=True)

        html_path = demo_path / f"{file.stem}.html"
        html_dest_path = dest_path / html_path.name

        subprocess.run(
            [
                JUPYTER,
                "nbconvert",
                "--to",
                "html",
                "--template",
                "lab",
                "--theme",
                "dark",
                str(file),
                ],
            shell=False,
            check=True
        )

        html_path.rename(html_dest_path)

    ### Run doxy ###

    subprocess.run(
        [
            DOXYGEN,
            str(doxy_path / "Doxyfile"),
            ],
        shell=False,
        check=True
        )

    ### Fix filelist ###
    
    filelist_path = doc_path / "files.html"

    if filelist_path.exists():
        # TODO better way to do this??
        content = filelist_path.read_text()
        content = content.replace('File List', 'Full Software Documentation')
        content = content.replace(
            "Here is a list of all documented files with brief descriptions:",
            "Here is a list containing the full software documentation. The structure of this page reflects the source code hierarchy."
        )
        filelist_path.write_text(content)

if __name__ == "__main__":
    check_deps()
    generate_docs()

