Thread control and MPI helpers
==============================

Thread control
--------------

Documentation for functions that control the global thread settings in ``daal4py``:

.. autofunction:: daal4py.daalinit
.. autofunction:: daal4py.num_threads
.. autofunction:: daal4py.enable_thread_pinning

MPI helpers
-----------

Documentation for helper functions that can be used in distributed mode, particularly when using MPI without ``mpi4py``:

.. autofunction:: daal4py.daalfini
.. autofunction:: daal4py.num_procs
.. autofunction:: daal4py.my_procid
