HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MPI=1 HOROVOD_WITHOUT_GLOO=1 HOROVOD_BUILD_ARCH_FLAGS=-L HOROVOD_CPU_OPERATIONS="SHMEM" HOROVOD_WITH_SHMEM=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_DEFAULT_CPU_OPERATIONS=S python3 setup.py clean
HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MPI=1 HOROVOD_WITHOUT_GLOO=1 HOROVOD_BUILD_ARCH_FLAGS=-L HOROVOD_CPU_OPERATIONS="SHMEM" HOROVOD_WITH_SHMEM=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_DEFAULT_CPU_OPERATIONS=S python3 setup.py sdist
HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MPI=1 HOROVOD_WITHOUT_GLOO=1 HOROVOD_BUILD_ARCH_FLAGS=-L HOROVOD_CPU_OPERATIONS="SHMEM" HOROVOD_WITH_SHMEM=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_DEFAULT_CPU_OPERATIONS=S pip3 install --no-cache-dir dist/horovod-0.19.2.tar.gz

