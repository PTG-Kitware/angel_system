=================================
Local Third-Party Python Packages
=================================

Some python dependencies are not published to a package manager so we must
include them in a custom manner.

As Git Submodules
=================
When a python package is available as a public git repository, we include them
as a git submodule underneath the ``./python-tpl/`` directory.

Adding a new Python TPL repo
----------------------------

Submodule
^^^^^^^^^
We of course need to add the package repo as a submodule:

.. code-block:: bash

   $ git submodule add -- REPO_URL_HERE ./python-tpl/name-of-package

We need to add to our python project dependencies to require this package.
This is important so that we can balance this new package's transitive
dependencies with the rest of our project's dependencies so that they are
compatible (as deemed by poetry's solver).

Poetry Dependency
^^^^^^^^^^^^^^^^^
We need to add the package to our poetry-based requirements.
We should do this inside the angel-workspace container
(``./angel-workspace-shell.sh``) so that we perform this addition with respect
to the existing environment and with the in-container version of poetry (this
is important for reproducible container builds).

For this, the python-tpl package should be pip-installable and have its runtime
dependencies correctly reflected via its ``setup.py``, or equivalent
configuration.

When inside the container:

.. code-block:: bash

   $ poetry add -e --lock ./python-tpl/yolov7/

* We use the ``-e`` option to mark the dependency as ``develop = true`` in the
  dependency configuration

* ``--lock`` only updates the :file:`poetry.lock` file without performing a
  local installation as we are preparing to build a new container, so a local
  environment modification is ephemeral and temporary.

It may be required to remove a root-owned ``{PACKAGE_NAME}.egg-info`` from the
root of the sub-module before proceeding.

Now, new a new docker container image may be built which will include
installing the package in an editable mode inside of the container environment.
