import pytest
import os
import subprocess
import tempfile

import nbformat

notebook_dir = os.path.join(os.path.split(__file__)[0], '..', 'notebook')


# https://blog.thedataincubator.com/2016/06/testing-jupyter-notebooks/
def _notebook_run(path):
    """Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    """
    dirname, __ = os.path.split(path)
    os.chdir(dirname)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=600",
                "--output", fout.name, path]
        subprocess.check_call(args)

        # fout.seek(0)
        nb = nbformat.read(fout.name, nbformat.current_nbformat)

    errors = [output
              for cell in nb.cells if "outputs" in cell
              for output in cell["outputs"]
              if output.output_type == "error"]

    return nb, errors


notebooks = [
    "Aspheric Newtonian Telescope.ipynb",
    "DECam perturbations.ipynb",
    "DECam trace.ipynb",
    "Distortion.ipynb",
    "HSC 3D.ipynb",
    "HSC perturbations.ipynb",
    "HSC trace.ipynb",
    "LSST donuts.ipynb",
    "LSST perturbations.ipynb",
    "LSST trace.ipynb",
    "Newtonian Telescope.ipynb",
    "PH != Pupil.ipynb",
    "Rays.ipynb",
    "Thin Lens.ipynb"
]

# We're purposely excluding the following for time:
# "FFT vs Huygens.ipynb"


@pytest.mark.timeout(600)
@pytest.mark.parametrize("notebook_name", notebooks)
def test_notebook(notebook_name):
    nb, errors = _notebook_run(os.path.join(notebook_dir, notebook_name))
    assert errors == []


if __name__ == '__main__':
    for notebook in notebooks:
        test_notebook(notebook)
