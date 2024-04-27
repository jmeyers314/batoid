import pytest
import os
from test_helpers import timer
from pathlib import Path


notebook_dir = Path(__file__).resolve().parent.parent / 'notebook'
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'


def _notebook_run(path):
    """Execute a notebook via nbclient and collect output.
       :returns (parsed nb object, execution errors)
    """
    import nbformat
    import nbclient
    # Load the notebook
    nb = nbformat.read(path, as_version=4)

    # Setup the client, specifying the timeout and the kernel
    client =  nbclient.NotebookClient(nb, timeout=300, kernel_name='python3')

    # Execute the notebook
    client.execute(cwd=path.parent)

    # Collect errors
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]

    return nb, errors


notebooks = [
    "Analytic despace aberration.ipynb",
    "Apochromat.ipynb",
    "Aspheric Newtonian Telescope.ipynb",
    "AuxTel trace.ipynb",
    "ComCam trace.ipynb",
    "DECam draw2d.ipynb",
    "DECam trace.ipynb",
    "DESI model details.ipynb",
    "DESI trace.ipynb",
    "Distortion.ipynb",
    "HSC pupil characterization.ipynb",
    "HSC trace.ipynb",
    "LSST+CBP.ipynb",
    "LSST trace.ipynb",
    "Newtonian Telescope.ipynb",
    "Thin Lens.ipynb",
    "PH != Pupil.ipynb"
]
slow_notebooks = [
    "AuxTel perturbations.ipynb",
    "DECam perturbations.ipynb",
    "FFT vs Huygens.ipynb",
    "Huygens PSF.ipynb",
    "HSC 3D.ipynb",
    "HSC ghosts.ipynb",
    "LSST donuts.ipynb",
    "LSST ghosts.ipynb",
    "LSST perturbations.ipynb",
    "LSST pupil characterization.ipynb",
    "HSC perturbations.ipynb"
]

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--slow', action='store_true')
    args = parser.parse_args()
    if args.slow:
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
    nb, errors = _notebook_run(notebook_dir / notebook_name)
    assert errors == []


if __name__ == '__main__':
    for notebook in notebooks:
        print(notebook)
        test_notebook(notebook)
