Installation Instructions
-------------------------

The `conda` Python distribution is recommended, but you can use any Python
distribution you see fit.

**Install `conda`**

.. seealso::

    Miniconda package management system:
    https://docs.conda.io/en/latest/miniconda.html


1. Install `miniconda` for your operating system:
2. We recommend to create a new conda environment:

.. code-block:: bash

    conda create -n twistpy python=3.9
    conda activate twistpy

.. note:: Make sure the `twistpy` environment is active when using `TwistPy` and for all the following steps on
    this page!

**Install TwistPy**

Clone TwistPy

.. code-block:: bash

    git clone https://github.com/solldavid/TwistPy.git

.. code-block:: bash

    cd TwistPy
    conda activate twistpy
    pip install -e .

**Update TwistPy**

To update `TwistPy` please change to the `TwistPy` directory and run

.. code-block:: bash

    git pull

If that does not work for some reason (e.g. the `TwistPy` repository has seen local changes, ...)
please do the following

.. code-block:: bash

    git fetch origin main
    git reset --hard origin/main

.. warning:: All your local changes will be deleted!

If the `TwistPy` dependencies changed, just run

.. code-block:: bash

    pip install -e .

again. Make sure the correct `conda` environment is active.