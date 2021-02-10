.. raw:: html

    <p align="center"><img src="https://user-images.githubusercontent.com/16640218/34506318-84d0c06c-efe0-11e7-8831-0425772ed8f2.png" alt="Logo" width="200"/></p>
    <br/>

Horovod
=======

.. image:: https://badge.buildkite.com/6f976bc161c69d9960fc00de01b69deb6199b25680a09e5e26.svg?branch=master
   :target: https://buildkite.com/horovod/horovod
   :alt: Build Status

.. image:: https://readthedocs.org/projects/horovod/badge/?version=latest
   :target: https://horovod.readthedocs.io/en/latest/
   :alt: Documentation Status

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :alt: License

.. image:: https://app.fossa.com/api/projects/git%2Bgithub.com%2Fhorovod%2Fhorovod.svg?type=shield
   :target: https://app.fossa.com/projects/git%2Bgithub.com%2Fhorovod%2Fhorovod?ref=badge_shield
   :alt: FOSSA Status

.. image:: https://bestpractices.coreinfrastructure.org/projects/2373/badge
   :target: https://bestpractices.coreinfrastructure.org/projects/2373
   :alt: CII Best Practices

.. image:: https://pepy.tech/badge/horovod
   :target: https://pepy.tech/project/horovod
   :alt: Downloads

.. inclusion-marker-start-do-not-remove

|

Horovod is a distributed deep learning training framework for TensorFlow, Keras, PyTorch, and Apache MXNet.
The goal of Horovod is to make distributed deep learning fast and easy to use.

This version of Horovod has SHMEM enabled.

Dependencies
-------
These are versions of packages/libraries that are necessary/known to work when using Shmorovod. Python packages are installed using ``pip``.

**Horovod**: ``0.19.2``

**OpenMPI**: ``4.0.3``

**OpenSHMEM**: ``1.4`` (included with **OpenMPI** >= ``4.x.x``)

**gcc**: ``8.0.3``

**torch**: ``1.7.0``

**torchvision**: ``0.8.0``

**tensorflow**: ``1.x.x``, ``2.0.0``

**h5py**: ``2.10.0``

**cffi**: 1.14.3``

**cloudpickle**: ``1.6.0``

Install
-------

1. Install `Open MPI <https://www.open-mpi.org/>`_ or another MPI implementation. Learn how to install Open MPI `on this page <https://www.open-mpi.org/faq/?category=building#easy-build>`_.

   **Note**: Open MPI 3.1.3 has an issue that may cause hangs. The recommended fix is to downgrade to Open MPI 3.1.2 or upgrade to Open MPI 4.0.0.

.. raw:: html

    <p/>

2. If you've installed TensorFlow from `PyPI <https://pypi.org/project/tensorflow>`__, make sure that the ``g++-4.8.5`` or ``g++-4.9`` is installed.

   If you've installed PyTorch from `PyPI <https://pypi.org/project/torch>`__, make sure that the ``g++-4.9`` or above is installed.

   If you've installed either package from `Conda <https://conda.io>`_, make sure that the ``gxx_linux-64`` Conda package is installed.

.. raw:: html

    <p/>

3. Install SHMEM-based **Horovod** from source.

   Download repository from GitHub.

   .. code-block:: bash

      $ git clone https://github.com/collinabidi/horovod
      
Build
-------

1. Enable **PyTorch** and/or **TensorFlow**.
   Modify the ``build_mpi.sh`` and ``build_shmem.sh`` scripts to include the proper flags. If you want to build with PyTorch, make sure that ``HOROVOD_WITH_PYTORCH=1`` is in each of the lines in ``build_mpi.sh`` and ``build_shmem.sh``. If you want to build with TensorFlow, make sure that ``HOROVOD_WITH_TENSORFLOW=1`` is in each of the lines in ``build_mpi.sh`` and ``build_shmem.sh``. If you want to build **without** one, add the ``HOROVOD_WITHOUT_TENSORFLOW=1`` or ``HOROVOD_WITHOUT_PYTORCH=1`` flags.

2. Build **Horovod** with **MPI** or **SHMEM**.

   To build Horovod with MPI enabled, run the ``build_mpi.sh`` script.

   .. code-block:: bash

      $ ./build_mpi.sh

   To build Horovod with SHMEM enabled, run the ``build_shmem.sh`` script.

   .. code-block:: bash

      $ ./build_shmem.sh
      
Usage
-------
1. Run **Horovod** with SLURM.

   If you use SLURM to submit jobs, simply modify the included SLURM script to fit you cluster's configuration. Make sure to correctly load your environment before executing anything. 

   .. code-block:: bash

      $ sbatch hvd_test_2.slurm
      
1. Run **Horovod** without SLURM.

   If you have admin access to your cluster, you can copy the SLURM script into a shell script, remove the variables at the top, and execute normally using the ``horovodrun`` or ``oshrun`` (SHMEM-specific) commands. 
   
   The following is an example of running the included ``pytorch_mnist.py`` script included in the ``example`` folder on 2 nodes (denoted by the ``-np 2`` argument). SHMEM-enabled version of Horovod has several necessary command-line arguments that may vary from system-to-system.

   .. code-block:: bash

      $ oshrun -np 2 -x --mca mpi_cuda_support 0 \
	--mca pml ucx --mca osc ucx \
	--mca atomic ucx --mca orte_base_help_aggregate 0 \
	--mca btl ^vader,tcp,openib,uct python3 pytorch_basic.py --epochs 1 --no-cuda
