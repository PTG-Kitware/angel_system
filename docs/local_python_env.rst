=====================================
Setting up a Local Python Environment
=====================================

Setting up ``pyenv``
====================
``pyenv`` is a tool to locally install various python versions separably
GitHub page is `here <https://github.com/pyenv/pyenv>`_ which includes
reference to the automatic installer script in `this section
<https://github.com/pyenv/pyenv#automatic-installer>`_.
The script that is downloaded and run can be viewed by following the links in
your browser if you would like to vet what is to be run on your system (a
recommended practice).

The following are additional packages to install on your host system to enable
building a fully capable Python installation.
The package names are in the form that would be found via the Ubuntu package
manager.

* ``libbz2-dev``

* ``libreadline-dev``

* ``tk-dev``

Creating Poetry-based Environment
=================================
A ``.python-version`` is present in the root of this repository that specifies
the python version that should be run by ``pyenv`` shims when within this
project's repository.
This version should be installed with ``pyenv`` so that we can install our
python dependencies locally in the next step.
Navigate to the root of this repository and run:

.. prompt::

   pyenv install $(cat .python-version)

With the python version successfully installed and available via ``pyenv``, we
can now create our python virtualenv via Poetry:

.. prompt::

   poetry install
