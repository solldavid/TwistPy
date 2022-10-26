Contributing
############

We happily welcome contributions of any kind! Please refer to the guide below on how to contribute.

Bug reports
===========

If you find any bugs, please report them at https://github.com/solldavid/TwistPy/issues.

When reporting a bug, please state the following:
* Your operating system name and version.
* Any details about your Python environment.
* A minimal example to reproduce the bug.

New feature suggestion
======================

Open an issue at https://github.com/solldavid/TwistPy/issues with tag *enhancement*.

* Explain in detail what your new feature should include.
* Keep the scope as narrow as possible, to make it easier to implement.

Add examples or improve documentation
=====================================

We are always happy to include new examples in the gallery!

Step-by-step instructions for contributing
******************************************

Ready to contribute?

1. Follow all instructions in :ref:`DevInstall`.

2. Create a branch for local development, usually starting from the dev branch:

.. code-block:: bash

   >> git checkout -b name-of-your-branch dev

Now you can make your changes locally.

3. When you're done making changes, check that your code follows the guidelines for :ref:`addingoperator` and
that the both old and new tests pass successfully:

.. code-block:: bash

   >> make tests

4. Run flake8 to check the quality of your code:

.. code-block:: bash

   >> make lint

Note that PyLops does not enforce full compliance with flake8, rather this is used as a
guideline and will also be run as part of our CI.
Make sure to limit to a minimum flake8 warnings before making a PR.

5. Update the docs

.. code-block:: bash

   >> make docupdate


6. Commit your changes and push your branch to GitHub:

.. code-block:: bash

   >> git add .
   >> git commit -m "Your detailed description of your changes."
   >> git push origin name-of-your-branch

Remember to add ``-u`` when pushing the branch for the first time.

7. Submit a pull request through the GitHub website.


Pull Request Guidelines
***********************

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include new tests for all the core routines that have been developed.
2. If the pull request adds functionality, the docs should be updated accordingly.
3. Ensure that the updated code passes all tests.

Project structure
#################
This repository is organized as follows:
* **twistpy**:      Python library
* **pytests**:      set of pytests
* **example_data**: sample datasets used in pytests and documentation
* **docs**:         Sphinx documentation
* **examples**:     Set of TwistPy examples to be embedded in documentation using sphinx-gallery
