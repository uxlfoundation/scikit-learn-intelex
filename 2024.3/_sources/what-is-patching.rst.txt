.. ******************************************************************************
.. * Copyright 2021 Intel Corporation
.. *
.. * Licensed under the Apache License, Version 2.0 (the "License");
.. * you may not use this file except in compliance with the License.
.. * You may obtain a copy of the License at
.. *
.. *     http://www.apache.org/licenses/LICENSE-2.0
.. *
.. * Unless required by applicable law or agreed to in writing, software
.. * distributed under the License is distributed on an "AS IS" BASIS,
.. * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. * See the License for the specific language governing permissions and
.. * limitations under the License.
.. *******************************************************************************/

.. _what_is_patching:

########
Patching
########

.. glossary::

   patching
      To patch scikit-learn with |intelex| is to replace stock scikit-learn algorithms
      with their optimized versions provided by the extension. You can always :term:`undo the patch <unpatching>`.

      There are different ways to patch scikit-learn:

      .. include:: /patching/patching-options.rst

   global patching
   
      .. include:: global-patching.rst
   
   unpatching
      To undo the patch is to return to the use of original scikit-learn implementation and
      replace patched algorithms with the stock scikit-learn algorithms.
      Unpatching requires scikit-learn to be re-imported again:

      .. include:: /patching/undo-patch.rst

How it works
------------

The extension replaces the original estimators in scikit-learn modules with the optimized ones.
If the desired algorithm parameters are not supported by the |intelex|,
then the result of the original scikit-learn is returned.

.. seealso:: :ref:`verbose`