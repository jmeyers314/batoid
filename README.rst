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

Batoid is known to work on MacOS and linux, using Python version 3.8+, and
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

If PyPI doesn't work, then you can try cloning the source from github and
running setup.py.  Clone the repo with either ::

    git clone git@github.com:jmeyers314/batoid.git

or ::

    git clone https://github.com/jmeyers314/batoid.git

Once the repo has been cloned, then compile and install with ::

    python setup.py install

or optionally ::

    python setup.py install --user

Tests
=====

To run the unit tests, from the batoid directory, first install the testing
requirements ::

    pip install -r test_requirements.txt

And then run the tests using pytest ::

    pytest
