Installation Guide
==================

Prerequisites
-------------

To begin, make sure that you have **Conda** installed on your system. Conda will help you manage dependencies and set up an isolated environment for this package.

1. **Install Conda**:
   If you don't have Conda installed yet, you can download and install it by following the instructions on the `Conda website <https://docs.anaconda.com/miniconda/>`_.

2. **Create and activate a new environment**:
   It's recommended to create a new environment for this project to avoid conflicts with other packages. Run the following commands:
   
   .. code-block:: bash

    conda create -n sargeom-env python=3.12
    conda activate sargeom-env

Installing the Package
----------------------

Once your environment is set up, you can install ``sargeom`` in one of three ways:

Option 1: Install from GitHub using pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install the package directly from GitHub using ``pip``:

.. code-block:: bash

    pip install git+https://github.com/oleveque/sargeom.git

Option 2: Install in Development Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to work on the package or use the latest developments from the repository, you can install it in development mode:

.. code-block:: bash

    git clone https://github.com/oleveque/sargeom.git
    pip install -r ./sargeom/requirements.txt
    pip install -e ./sargeom

Option 3: Add to `requirements.txt`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you're managing your project's dependencies using a `requirements.txt` file, simply add the following line:

.. code-block:: bash

    sargeom @ git+https://github.com/oleveque/sargeom@main

This will allow you to install ``sargeom`` by running:

.. code-block:: bash

    pip install -r requirements.txt
