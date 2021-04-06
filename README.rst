.. raw:: html

    <p align="center"><img src="https://user-images.githubusercontent.com/16640218/34506318-84d0c06c-efe0-11e7-8831-0425772ed8f2.png" alt="Logo" width="200"/></p>
    <br/>

Horovod
=======

.. raw:: html

   <div align="center">

.. image:: https://badge.fury.io/py/horovod.svg
   :target: https://badge.fury.io/py/horovod
   :alt: PyPI Version

.. image:: https://badge.buildkite.com/6f976bc161c69d9960fc00de01b69deb6199b25680a09e5e26.svg?branch=master
   :target: https://buildkite.com/horovod/horovod
   :alt: Build Status

.. image:: https://readthedocs.org/projects/horovod/badge/?version=latest
   :target: https://horovod.readthedocs.io/en/latest/
   :alt: Documentation Status

.. image:: https://img.shields.io/badge/slack-chat-green.svg?logo=slack
   :target: https://forms.gle/cPGvty5hp31tGfg79
   :alt: Slack

.. raw:: html

   </div>

.. raw:: html

   <div align="center">

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

.. raw:: html

   </div>

.. inclusion-marker-start-do-not-remove

|

Horovod is a distributed deep learning training framework for TensorFlow, Keras, PyTorch, and Apache MXNet.
The goal of Horovod is to make distributed deep learning fast and easy to use.

This version of Horovod has SHMEM enabled and was built by Collin Abidi at NSF SHREC, University of Pittsburgh.

Dependencies
-------
These are versions of packages/libraries that are necessary/known to work when using Shmorovod. Python packages are installed using ``pip``.
.. raw:: html

   <p><img src="https://raw.githubusercontent.com/lfai/artwork/master/lfaidata-assets/lfaidata-project-badge/graduate/color/lfaidata-project-badge-graduate-color.png" alt="LF AI & Data" width="200"/></p>


Horovod is hosted by the `LF AI & Data Foundation <https://lfdl.io>`_ (LF AI & Data). If you are a company that is deeply
committed to using open source technologies in artificial intelligence, machine, and deep learning, and want to support
the communities of open source projects in these domains, consider joining the LF AI & Data Foundation. For details
about who's involved and how Horovod plays a role, read the Linux Foundation `announcement <https://lfdl.io/press/2018/12/13/lf-deep-learning-welcomes-horovod-distributed-training-framework-as-newest-project/>`_.

|

.. contents::

|

Documentation
-------------

**Horovod**: ``0.19.2``

**OpenMPI**: ``4.0.3``

**UCX**

**OpenSHMEM**: ``1.4`` (included with **OpenMPI** >= ``4.0``)

**gcc**: ``8.0.3``

**torch**: ``1.7.0``

**torchvision**: ``0.8.0``

**tensorflow**: ``1.x.x``, ``2.0.0``

**h5py**: ``2.10.0``

**cffi**: ``1.14.3``

**cloudpickle**: ``1.6.0``

Install
-------

1. Install `CMake <https://cmake.org/install/>`__

.. raw:: html

    <p/>

2. If you've installed TensorFlow from `PyPI <https://pypi.org/project/tensorflow>`__, make sure that the ``g++-4.8.5`` or ``g++-4.9`` or above is installed.

   If you've installed PyTorch from `PyPI <https://pypi.org/project/torch>`__, make sure that the ``g++-4.9`` or above is installed.

   If you've installed either package from `Conda <https://conda.io>`_, make sure that the ``gxx_linux-64`` Conda package is installed.

.. raw:: html

    <p/>
3. Install OpenMPI and UCX with OpenMPI following the instructions at https://github.com/openucx/ucx/wiki/OpenMPI-and-OpenSHMEM-installation-with-UCX.

4. Install SHMEM-based **Horovod** from source.

   Download repository from GitHub.

   .. code-block:: bash

      $ git clone --recursive https://github.com/collinabidi/horovod
      
Build
-------

1. Enable **PyTorch** and/or **TensorFlow**.
   Modify the ``build_mpi.sh`` and ``build_shmem.sh`` scripts to include the proper flags. If you want to build with PyTorch, make sure that ``HOROVOD_WITH_PYTORCH=1`` is in each of the lines of ``build_mpi.sh`` and ``build_shmem.sh``. If you want to build with TensorFlow, make sure that ``HOROVOD_WITH_TENSORFLOW=1`` is in each of the lines in ``build_mpi.sh`` and ``build_shmem.sh``. If you want to build **without** one, add the ``HOROVOD_WITHOUT_TENSORFLOW=1`` or ``HOROVOD_WITHOUT_PYTORCH=1`` flags.
      $ HOROVOD_GPU_OPERATIONS=NCCL pip install horovod

For more details on installing Horovod with GPU support, read `Horovod on GPU <docs/gpus.rst>`_.

For the full list of Horovod installation options, read the `Installation Guide <docs/install.rst>`_.

If you want to use MPI, read `Horovod with MPI <docs/mpi.rst>`_.

If you want to use Conda, read `Building a Conda environment with GPU support for Horovod <docs/conda.rst>`_.

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
      
2. Run **Horovod** without SLURM.

   If you have admin access to your cluster, you can copy the SLURM script into a shell script, remove the variables at the top, and execute normally using the ``horovodrun`` or ``oshrun`` (SHMEM-specific) commands. 
   
   The following is an example of running the included ``pytorch_mnist.py`` script included in the ``example`` folder on 2 nodes (denoted by the ``-np 2`` argument). SHMEM-enabled version of Horovod has several necessary command-line arguments that may vary from system-to-system.

   .. code-block:: bash

      $ oshrun -np 2 -x --mca mpi_cuda_support 0 \
	--mca pml ucx --mca osc ucx \
	--mca atomic ucx --mca orte_base_help_aggregate 0 \
	--mca btl ^vader,tcp,openib,uct python3 pytorch_basic.py --epochs 1 --no-cuda
       $ horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python train.py

3. To run using Open MPI without the ``horovodrun`` wrapper, see `Running Horovod with Open MPI <docs/mpirun.rst>`_.

4. To run in Docker, see `Horovod in Docker <docs/docker.rst>`_.

5. To run in Kubernetes, see `Kubeflow <https://github.com/kubeflow/examples/tree/master/demos/yelp_demo/ks_app/vendor/kubeflow/mpi-job>`_, `MPI Operator <https://github.com/kubeflow/mpi-operator/>`_, `Helm Chart <https://github.com/kubernetes/charts/tree/master/stable/horovod/>`_, `FfDL <https://github.com/IBM/FfDL/tree/master/etc/examples/horovod/>`_, and `Polyaxon <https://docs.polyaxon.com/integrations/horovod/>`_.

6. To run on Spark, see `Horovod on Spark <docs/spark.rst>`_.

7. To run on Ray, see `Horovod on Ray <docs/ray.rst>`_.

8. To run in Singularity, see `Singularity <https://github.com/sylabs/examples/tree/master/machinelearning/horovod>`_.

9. To run in a LSF HPC cluster (e.g. Summit), see `LSF <docs/lsf.rst>`_.


Guides
------
1. Run distributed training in Microsoft Azure using `Batch AI and Horovod <https://github.com/Azure/BatchAI/tree/master/recipes/Horovod>`_.
2. `Distributed model training using Horovod <https://spell.ml/blog/distributed-model-training-using-horovod-XvqEGRUAACgAa5th>`_.

Send us links to any user guides you want to publish on this site

Troubleshooting
---------------
See `Troubleshooting <docs/troubleshooting.rst>`_ and submit a `ticket <https://github.com/horovod/horovod/issues/new>`_
if you can't find an answer.


Citation
--------
Please cite Horovod in your publications if it helps your research:

::

    @article{sergeev2018horovod,
      Author = {Alexander Sergeev and Mike Del Balso},
      Journal = {arXiv preprint arXiv:1802.05799},
      Title = {Horovod: fast and easy distributed deep learning in {TensorFlow}},
      Year = {2018}
    }


Publications
------------
1. Sergeev, A., Del Balso, M. (2017) *Meet Horovod: Uber’s Open Source Distributed Deep Learning Framework for TensorFlow*.
Retrieved from `https://eng.uber.com/horovod/ <https://eng.uber.com/horovod/>`_

2. Sergeev, A. (2017) *Horovod - Distributed TensorFlow Made Easy*. Retrieved from
`https://www.slideshare.net/AlexanderSergeev4/horovod-distributed-tensorflow-made-easy <https://www.slideshare.net/AlexanderSergeev4/horovod-distributed-tensorflow-made-easy>`_

3. Sergeev, A., Del Balso, M. (2018) *Horovod: fast and easy distributed deep learning in TensorFlow*. Retrieved from
`arXiv:1802.05799 <https://arxiv.org/abs/1802.05799>`_


References
----------
The Horovod source code was based off the Baidu `tensorflow-allreduce <https://github.com/baidu-research/tensorflow-allreduce>`_
repository written by Andrew Gibiansky and Joel Hestness. Their original work is described in the article
`Bringing HPC Techniques to Deep Learning <http://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/>`_.

Getting Involved
----------------
- `Community Slack <https://forms.gle/cPGvty5hp31tGfg79>`_ for collaboration and discussion
- `Horovod Announce <https://lists.lfai.foundation/g/horovod-announce>`_ for updates on the project
- `Horovod Technical-Discuss <https://lists.lfai.foundation/g/horovod-technical-discuss>`_ for public discussion


.. inclusion-marker-end-do-not-remove
   Place contents above here if they should also appear in read-the-docs.
   Contents below are already part of the read-the-docs table of contents.
