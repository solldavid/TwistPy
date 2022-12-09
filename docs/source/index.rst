.. TwistPy documentation master file, created by David Sollberger

Home
----

.. image:: _static/logo_adobe_title.png
    :alt: TwistPy Logo

TwistPy is a small open-source Python package for seismic data processing. It includes routines for single-station
polarization analysis and filtering, as well as array processing tools.

A special focus lies on innovative techniques to process spatial wavefield gradient data and, in particular,
rotational seismic data obtained from dedicated rotational seismometers or small-aperture arrays of three-component
sensors.

Some of the tools available in TwistPy are:

- Three-component polarization analysis and filtering (both time domain and S-transform).
- Six-component polarization analysis and filtering (both time domain and S-transform).
- Six-component wave type fingerprinting.
- Single-station six-component Love- and Rayleigh-wave dispersion and Rayleigh wave ellipticity angle estimation.
- Dynamic tilt corrections for seismometers using direct rotation measurements.
- Beamforming (Bartlett, MVDR, and MUSIC algorithm).
- Forward and inverse S-transform (Stockwell transform).

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: General information

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Instructions:

    installation.rst
    examples/index.rst
    contributing.rst

.. toctree::
    :maxdepth: 3
    :hidden:
    :caption: Reference documentation:

    api/index.rst

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: References:

    references.rst
    contributors.rst
    contact.rst
    acknowledgments.rst


