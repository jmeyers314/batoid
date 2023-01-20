import pytest
import os
import subprocess
import tempfile
from test_helpers import timer

import nbformat

notebook_dir = os.path.abspath(os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..',
    'notebook'
))


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
    "Analytic despace aberration.ipynb",
    "Apochromat.ipynb",
    "Aspheric Newtonian Telescope.ipynb",
    "AuxTel perturbations.ipynb",
    "AuxTel trace.ipynb",
    "ComCam trace.ipynb",
    "DECam draw2d.ipynb",
    "DECam perturbations.ipynb",
    "DECam trace.ipynb",
    "DESI model details.ipynb",
    "DESI trace.ipynb",
    "Distortion.ipynb",
    "HSC 3D.ipynb",
    "HSC pupil characterization.ipynb",
    "HSC trace.ipynb",
    "LSST+CBP.ipynb",
    "LSST donuts.ipynb",
    "LSST ghosts.ipynb",
    "LSST pupil characterization.ipynb",
    "LSST trace.ipynb",
    "Newtonian Telescope.ipynb",
    # "Rays.ipynb",
    "Thin Lens.ipynb",
    "PH != Pupil.ipynb"
]
slow_notebooks = [
    "FFT vs Huygens.ipynb",
    "Huygens PSF.ipynb",
    "HSC ghosts.ipynb",
    "LSST perturbations.ipynb",
    "HSC perturbations.ipynb"
]

if __name__ == '__main__':
    notebooks += slow_notebooks
else:
    notebooks = [
        pytest.param(notebook, marks=pytest.mark.skip_gha)
        for notebook in notebooks
    ]
    notebooks += [
        pytest.param(notebook, marks=[pytest.mark.skip_gha, pytest.mark.slow])
        for notebook in slow_notebooks
    ]


@pytest.mark.timeout(300)
@pytest.mark.parametrize("notebook_name", notebooks)
@timer
def test_notebook(notebook_name):
    nb, errors = _notebook_run(os.path.join(notebook_dir, notebook_name))
    assert errors == []


if __name__ == '__main__':
    for notebook in notebooks:
        test_notebook(notebook)
