.. image:: https://travis-ci.org/jmeyers314/batoid.svg?branch=master
        :target: https://travis-ci.org/jmeyers314/batoid
.. image:: https://codecov.io/gh/jmeyers314/batoid/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/jmeyers314/batoid
.. image:: https://readthedocs.org/projects/batoid/badge/?version=latest


batoid
======

A c++ backed python optical raytracer.


Requirements
============

Batoid is known to work on MacOS and linux, using Python version 3.4+, and
either the clang or gcc compiler with support for c++11.

Installation
============

This *should* be as simple as cloning the repo and running setup.py.  One minor
hiccup is that this repo includes submodules for ``pybind11`` and ``eigen``, so
when cloning for the first time, a command similar to one of the following
should be used ::

    git clone --recurse-submodules git@github.com:jmeyers314/batoid.git

or ::

    git clone --recurse-submodules https://github.com/jmeyers314/batoid.git

Once the repo and the submodules have been cloned, then compile and install with ::

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
