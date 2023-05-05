Contributing
############

We happily welcome contributions of any kind! Please refer to the guide below on how to contribute.

Bug reports
===========

If you find any bugs, please report them at https://github.com/solldavid/TwistPy/issues.

When reporting a bug, please include the following information:

*  TwistPy version, Python version, operating system.
*  If this is a new bug that did not exist in previous versions, please state in which version it last worked.
*  If possible, provide a minimal example to reproduce the bug.

New feature suggestion
======================

Open an issue at https://github.com/solldavid/TwistPy/issues with tag *enhancement*.

*  Explain in detail what your new feature should include.
*  Keep the scope as narrow as possible, to make it easier to implement.

Add examples or improve documentation
=====================================

We are always happy to include new examples in the gallery and improve our documentation!

Adding new features: Step-by-step instructions for developers
=============================================================

1. Fork the repository.

2. If you have not done so already, install ``pre-commit``, ``black`` and ``flake8``:

.. code-block:: bash

   >> pip install pre-commit black flake8

3. Install pre-commit hooks. The repository comes with a pre-commit configuration to reformat the code with ``black`` and check the code with ``flake8`` before each commit.

.. code-block:: bash

   >> pre-commit install

4. Create a new branch for local development, based at the main branch:

.. code-block:: bash

   >> git checkout -b name-of-your-branch main

5. Now you can make your changes locally.

6. Add a test for your changes.

7. Make sure that all tests pass using pytest.

8. Add your name to the list of contributors in ``/docs/source/contributors.rst``.

9. Push to your fork.

.. code-block:: bash

   >> git add .
   >> git commit -m "Commit message"
   >> git push origin name-of-your-branch

Remember to add ``-u`` when pushing the branch for the first time.

10. Submit a pull request via the GitHub website.

Continuous Integration
======================

CI is implemented with GitHub Actions with workflows that are run upon each commit to the repository for testing, linting, and documentation building.

Style Guide
===========

1. We use a default line length of 88 characters, which is the default of the ``black`` formatter. Note that this line length is not enforced for docstrings.
2. Source code must follow the PEP8 coding standards.
3. For better readability, docstrings need to be in `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_ format. Please refer to the numpydoc style guide!
4. Use type hints whenever possible!
