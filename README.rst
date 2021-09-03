.. image:: https://github.com/jmeyers314/batoid/workflows/batoid%20CI/badge.svg
        :target: https://github.com/jmeyers314/batoid/workflows/batoid%20CI/badge.svg
.. image:: https://codecov.io/gh/jmeyers314/batoid/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/jmeyers314/batoid


batoid
======

A c++ backed python optical raytracer.

docs
====
https://jmeyers314.github.io/batoid/overview.html


Requirements
============

Batoid is known to work on MacOS and linux, using Python version 3.6+, and
either the clang or gcc compiler with support for c++14.

Installation
============

PyPI
----

Released versions of batoid are available on pypi as source distributions.
This means you will need at least c++14 compiler available and that setup.py
can find it.  This should hopefully be the case on most *nix systems, in which
case, the following ought to work::

    pip install batoid

Github
------

If Pypi doesn't work, then you can try cloning the source from github and
running setup.py.  One minor hiccup in this case is that the batoid repo
includes ``pybind11`` as a submodule, so when cloning for the first time, a
command similar to one of the following should be used ::

    git clone --recurse-submodules git@github.com:jmeyers314/batoid.git

or ::

    git clone --recurse-submodules https://github.com/jmeyers314/batoid.git

Once the repo and the submodules have been cloned, then compile and install
with ::

    python setup.py install

or optionally ::

    python setup.py install --user

Tests
=====

To run the unit tests, from the batoid directory, first install the testing
requirements ::

    pip install -r test_requirements.txt

And then run the tests using setup.py ::

    python setup.py test
