# ==============================================================================
# Copyright 2014 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

###############################################################################
# The code generator.
# We define jinja2 templates to generate code for all the oneDAL algorithms and their
# Result and Model objects. Most macros work on one namespace and expect expect
# values/variables in their env. If not specificied otherwise, we assume we can
# use the following
#          {{ns}}:            current C++ namespace
#          {{algo}}:          algo name (as seen in API)
#          {{args_decl}}:     non-template arguments for wrapper function
#          {{args_call}}:     non-template arguments to pass on
#          {{input_args}}:    args for setting input
#          {{template_decl}}: template parameters for declaration
#          {{template_args}}: template arguments mapping to their possible values
#          {{params_req}}:    dict of required parameters and their values
#          {{params_opt}}:    dict of parameters and their values
#          {{params_get}}:    parameter struct getter
#          {{enum_vals}}:     list of tuples (name, enum_val) for using
#                             in distributed algos
#          {{step_specs}}:    distributed spec
#          {{map_result}}:    Result enum id for getting
#                             partial result (can be FULLPARTIAL)
#          {{iface}}:         Interface manager
#          {{result_map}}:    Type information about result type of given algo
#
# The code here is of course highly dependent on manual code that can be found in ../src.
# For example, we pretend we know how distributed algorithm interfaces look like and work.
#
# Also, we use some functions which accept
# PyObject* to convert the python objects manually
# Similarly, we create PyObjects manually in some C++ functions (e.g. nd-arrays).
###############################################################################
# FIXME remove remaining args/code for distributed computation if none available
# FIXME a revision from scratch would be helpful...

import re

import jinja2
from jinja2 import select_autoescape

from .wrappers import hpat_types

###############################################################################
# generic utility functions/defs needed by generated code
cython_header = """
# distutils: language = c++
#cython: language_level=2

# Import the Python-level symbols of numpy
import numpy as np

import sys

# Import the C-level symbols of numpy
cimport numpy as npc
# import std::string support
from libcpp.string cimport string as std_string
from libcpp cimport bool
from cpython.ref cimport PyObject
from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t

try:
    import pandas
    pdDataFrame = pandas.DataFrame
    pdSeries = pandas.Series
except ImportError:
    class pdDataFrame:
        pass
    class pdSeries:
        pass

npc.import_array()

hpat_spec = []

cdef extern from "daal4py_cpp.h":
    cdef cppclass NumericTablePtr:
        pass

    cdef cppclass data_management_DataCollectionPtr:
        pass

    ctypedef NumericTablePtr data_management_NumericTablePtr


cdef extern from "pickling.h":
    cdef object serialize_si(void *) nogil
    cdef T* deserialize_si[T](object) nogil


cpdef _rebuild(class_constructor, state_data):
    cls = class_constructor()
    cls.__setstate__(state_data)
    return cls


cdef class data_management_datacollection:
    cdef data_management_DataCollectionPtr * c_ptr

    def __dealloc__(self):
        del self.c_ptr


cdef extern from "daal4py.h":
    cdef double get_nan64()
    cdef float get_nan32()

    cdef cppclass data_or_file :
        data_or_file(PyObject *) except +

    cdef cppclass list_NumericTablePtr:
        pass

    cdef cppclass dict_NumericTablePtr:
        pass

    cdef std_string to_std_string(PyObject * o) except +

    cdef object make_nda(NumericTablePtr * nt_ptr) except +
    cdef object make_nda(list_NumericTablePtr * nt_ptr) except +
    cdef object make_nda(dict_NumericTablePtr * nt_ptr, void *) except +
    cdef NumericTablePtr make_nt(PyObject * nda) except +
    cdef list_NumericTablePtr make_datacoll(PyObject * nda) except +
    cdef dict_NumericTablePtr make_dnt(PyObject * nda, void *) except +

    cdef T* dynamicPointerPtrCast[T,U](U*)
    cdef bool is_valid_ptrptr[T](T * o)

    cdef void * e2s_algorithms_pca_result_dataForTransform
    cdef void * s2e_algorithms_pca_transform

    cdef T* _daal_clone[T](const T & o)

NAN64 = get_nan64()
NAN32 = get_nan64()


cdef extern from "daal.h":
    ctypedef unsigned long long DAAL_UINT64


cdef extern from "daal4py_cpp.h":
    cdef void c_daalinit(int nthreads) except +
    cdef void c_daalfini() except +
    cdef size_t c_num_threads() except +
    cdef size_t c_num_procs() except +
    cdef size_t c_my_procid() except +
    cdef void c_enable_thread_pinning(bool enabled) except +


def daalinit(nthreads: int = -1) -> None:
    '''
    Set number of threads for daal4py

    This modifies the number of threads configured for daal4py, which is a
    global setting - meaning: it is applied to all subsequent calls to daal4py
    functions / methods in the Python process.

    By default, if not otherwise configured, it will use the full number of threads
    available in the system.

    :param int nthreads: [default: -1] Number of threads to use for further computations in daal4py. If this number is less or equal than zero, then settings will not be changed.

    :rtype: None
    '''
    c_daalinit(nthreads)

def daalfini() -> None:
    '''
    Finalize MPI environment

    When using distributed mode without ``mpi4py``, this function must be called after
    the distributed computation calls before accessing the result object from the algorithm
    that was executed in distributed mode. It has no effect when the python process is not
    run through MPI (used for distributed mode).

    This is a wrapper over ``MPI_Finalize``. It does not need to be called if ``mpi4py``
    was imported before, as ``mpi4py`` calls this function upon process exit.

    Note that software ``mpi4py`` calls this function automatically if it is imported, but
    it only does so upon process exit, so this still needs to be called before accessing
    the result objects in the process/rank that will use them.

    :rtype: None
    '''
    c_daalfini()

def num_threads() -> int:
    '''
    Gets number of threads configured for daal4py.

    .. note:: The number of threads for daal4py is a global setting, which can be changed through :obj:`daalinit`.

    :rtype: int
    '''
    return c_num_threads()

def num_procs() -> int:
    '''
    Get number of MPI processes (in distributed mode)

    If the python process is not run though MPI, this function will always return 1.

    This is a wrapper over ``MPI_Comm_size``. Equivalent to :obj:`mpi4py.MPI.Comm.Get_size`,
    but does not require ``mpi4py`` to be installed.

    :rtype: int
    '''
    return c_num_procs()

def my_procid() -> int:
    '''
    Get MPI process rank

    If the python process is not being run through MPI (used for distributed mode), this
    will always return zero.

    This is a wrapper over ``MPI_Comm_rank``. Equivalent to :obj:`mpi4py.MPI.Comm.Get_rank`,
    but does not require ``mpi4py`` to be installed.

    :rtype: int
    '''
    return c_my_procid()

def enable_thread_pinning(enabled: bool = True) -> None:
    '''
    Enable or disable thread pinning

    This function enables or disables binding of the threads that are used to parallelize
    algorithms of the library to physical processing units for
    possible performance improvement. Improper use of the method can
    result in degradation of the application performance depending on
    the system (machine) topology, application, and operating system.
    By default, pinning is disabled.

    .. note:: This is a global setting for daal4py.

    :param bool enabled: [default: True] Whether to enable thread pinning or not.

    :rtype: None
    '''
    c_enable_thread_pinning(enabled)

def get_data(x):
    if isinstance(x, pdDataFrame):
        x_dtypes = x.dtypes.values
        if np.all(x_dtypes == x_dtypes[0]):
            x = x.to_numpy()
        else:
            x = [xi.to_numpy() for _, xi in x.items()]

    elif isinstance(x, pdSeries):
        x = x.to_numpy().reshape(-1, 1)

    elif "modin" in sys.modules:
        from modin import pandas as mdpd
        if isinstance(x, mdpd.DataFrame):
            x = x.to_numpy()
        elif isinstance(x, mdpd.Series):
            x = x.to_numpy().reshape(-1, 1)
    return x


def _str(instance, properties):
        result = ''
        for p in properties:
            prop = getattr(instance, p, None)
            tail = ''
            if isinstance(prop, np.ndarray):
                result += p + ': array('
                tail = ',\\n  dtype={}, shape={})'.format(prop.dtype,
                                                        prop.shape)
            elif isinstance(prop, dict):
                result += p + ': dict, len={}'.format(len(prop))
            else:
                result += p + ': '
            value = '\\n  '.join(str(prop).splitlines())
            if '\\n' in value or isinstance(prop, np.ndarray):
                result += '\\n  '
            result += value
            result += tail
            result += '\\n\\n'
        return result[:-2]


cdef extern from "daal4py.h":
    cdef bool c_assert_all_finite(const data_or_file & t,
                                  bool allowNaN, char dtype) except +


def daal_assert_all_finite(X, allow_nan=False, dtype=0):
    return c_assert_all_finite(data_or_file(<PyObject*>X), allow_nan, dtype)


cdef extern from "daal4py.h":
    cdef void c_train_test_split(data_or_file & orig, data_or_file & train,
                                 data_or_file & test, data_or_file & train_idx,
                                 data_or_file & test_idx) except +


def daal_train_test_split(orig, train, test, train_idx, test_idx):
    c_train_test_split(data_or_file(<PyObject*>orig), data_or_file(<PyObject*>train),
                       data_or_file(<PyObject*>test), data_or_file(<PyObject*>train_idx),
                       data_or_file(<PyObject*>test_idx))


cdef extern from "daal4py.h":
    cdef double c_roc_auc_score(data_or_file & y_true, data_or_file & y_test) except +


def daal_roc_auc_score(y_true, y_test):
    return c_roc_auc_score(data_or_file(<PyObject*>y_true),
                           data_or_file(<PyObject*>y_test))

cdef extern from "daal4py.h":
    cdef void c_generate_shuffled_indices(data_or_file & idx,
                                          data_or_file & random_state) except +


def daal_generate_shuffled_indices(idx, random_state):
    c_generate_shuffled_indices(data_or_file(<PyObject*>idx),
                                data_or_file(<PyObject*>random_state))


cdef extern from "daal4py.h":
    cdef void c_tsne_gradient_descent(data_or_file & init, data_or_file & p,
                                      data_or_file & size_iter, data_or_file & params,
                                      data_or_file & results, char dtype) except +


def daal_tsne_gradient_descent(init, p, size_iter, params, results, dtype=0):
    c_tsne_gradient_descent(data_or_file(<PyObject*>init), data_or_file(<PyObject*>p),
                            data_or_file(<PyObject*>size_iter),
                            data_or_file(<PyObject*>params),
                            data_or_file(<PyObject*>results), dtype)
"""

###############################################################################
# generates result/model classes
# Generates header, c++ and cython code together, separated by "%SNIP%
# requires {{enum_gets}}    list of triplets of members accessed via get(ns, name, type)
#          {{named_gets}}   list of pairs(name, type) of
#                           members accessed via type var = getName()
#          {{class_type}}   C++ class
typemap_wrapper_template = """
{% set flatname = (class_type|flat|strip(' *')|lower).replace('ptr', '') %}
{% set splitname = class_type.rsplit('::', 1) %}
typedef {{class_type}} {{class_type|flat|strip(' *')}};
{% if enum_gets or named_gets %}
{% for m in enum_gets %}
extern "C" {{m[2]}} {{'*' if 'Ptr' in m[2] else ''}} get_{{flatname}}_{{m[1]}}\
({{class_type}} * obj_);
{% endfor %}
{% for m in named_gets %}
extern "C" {{m[0]}} {{'*' if 'Ptr' in m[0] else ''}} get_{{flatname}}_{{m[1]}}\
({{class_type}} * obj_);
{% endfor %}
{% for m in get_methods %}
extern "C" {{m[0]}} * get_{{flatname}}_{{m[1]}}({{class_type}} *, {{m[2]}});
{% endfor %}
{% if (flatname.startswith('gbt_') or \
    flatname.startswith('decision_forest')) and flatname.endswith('model') %}
// FIXME
extern "C" size_t get_{{flatname}}_numberOfTrees({{class_type}} * obj_);
{% endif %}
%SNIP%
{% for m in enum_gets %}
extern "C" {{m[2]}} {{'*' if 'Ptr' in m[2] else ''}} get_{{flatname}}_{{m[1]}}\
({{class_type}} * obj_)
{
    return RAW< {{m[2]}} >()((*obj_)->get(daal::{{m[0]}}::{{m[1]}}));
}
{% endfor %}
{% for m in named_gets %}
extern "C" {{m[0]}} {{'*' if 'Ptr' in m[0] else ''}} get_{{flatname}}_{{m[1]}}\
({{class_type}} * obj_)
{
    return RAW< {{m[0]}} >()((*obj_)->get{{m[1]}}{{m[2]}}());
}
{% endfor %}
{% for m in get_methods %}
extern "C" {{m[0]}} * get_{{flatname}}_{{m[1]}}({{class_type}} * obj_, {{m[2]}} {{m[3]}})
{
    return new {{m[0]}}((*obj_)->get{{m[1]}}({{m[3]}}));
}
{% endfor %}
{% if (flatname.startswith('gbt_') or \
    flatname.startswith('decision_forest')) and flatname.endswith('model') %}
// FIXME
extern "C" size_t get_{{flatname}}_numberOfTrees({{class_type}} * obj_)
{
    return (*obj_)->numberOfTrees();
}
{% endif %}

%SNIP%
cdef extern from "daal4py_cpp.h":
    cdef cppclass {{class_type|flat|strip(' *')}}:
        pass

{% if not class_type.startswith('daal::'+ns) %}
{% endif %}
cdef extern from "daal4py_cpp.h":
{% for m in enum_gets %}
    cdef {{m[2]|d2cy}} get_{{flatname}}_{{m[1]}}({{class_type|flat}} obj_) except +
{% endfor %}
{% for m in named_gets %}
    cdef {{m[0]|d2cy}} get_{{flatname}}_{{m[1]}}({{class_type|flat}} obj_) except +
{% endfor %}
{% for m in get_methods %}
    cdef {{(m[0]|d2cy)}} get_{{flatname}}_{{m[1]}}({{class_type|flat}} \
        obj_, {{m[2]}} {{m[3]}}) except +
{% endfor %}
{% if (flatname.startswith('gbt_') or \
    flatname.startswith('decision_forest')) and flatname.endswith('model') %}
    # FIXME
    cdef size_t get_{{flatname}}_numberOfTrees({{class_type|flat}} obj_) except +
{% endif %}

cdef class {{flatname}}:
    '''
    Properties:
    '''
    cdef {{class_type|flat}} c_ptr

    def __cinit__(self):
        self.c_ptr = NULL
        pass

    def __dealloc__(self):
        del self.c_ptr

    def __init__(self, int64_t ptr=0):
        self.c_ptr = <{{class_type|flat}}>ptr

    def __repr__(self):
        return _str(self, [{% for m in enum_gets+named_gets %}'{{m[1]}}',{% endfor %}])
{% for m in enum_gets+named_gets %}
{% set rtype = m[2]|d2cy(False) if m in enum_gets else m[0]|d2cy(False) %}

    @property
    def {{m[1]}}(self):
{% if ('Ptr' in rtype and 'NumericTablePtr' not in rtype) or '__iface__' in rtype %}
{% set frtype=(rtype.strip(' *&')|flat(False)|strip(' *')).replace('Ptr', '')|lower %}
        '''
        {{m[1]}}
        :type: {{frtype.replace('data_management_NumericTablePtr', 'Numpy array')}}
        '''
        if not is_valid_ptrptr(self.c_ptr):
            raise ValueError("Pointer to oneDAL entity is NULL")
        cdef {{frtype}} res = {{frtype}}.__new__({{frtype}})
        res.c_ptr = get_{{flatname}}_{{m[1]}}(self.c_ptr)
        return res
{% else %}
        '''
        {{m[1]}}
        :type: {{'Numpy array' if 'NumericTablePtr' in rtype else rtype}}
        '''
        if not is_valid_ptrptr(self.c_ptr):
            raise ValueError("Pointer to oneDAL entity is NULL")
{% if 'NumericTablePtr' in rtype %}
        cdef {{rtype}} * res = get_{{flatname}}_{{m[1]}}(self.c_ptr)
        res_obj = {{'<object>make_nda(res, e2s_algorithms_'+flatname+'_'+m[1]+')' \
            if 'dict_NumericTablePtr' in rtype else '<object>make_nda(res)'}}
        del res
        return res_obj
{% else %}
        res = get_{{flatname}}_{{m[1]}}(self.c_ptr)
        return res
{% endif %}
{% endif %}
{% endfor %}

{% if (flatname.startswith('gbt_') or flatname.startswith('decision_forest')) \
    and flatname.endswith('model') %}
    @property
    def NumberOfTrees(self):
        '''
        NumberOfTrees
        :type: size_t
        '''
        if not is_valid_ptrptr(self.c_ptr):
            raise ValueError("Pointer to oneDAL entity is NULL")
        return get_{{flatname}}_numberOfTrees(self.c_ptr)
{% endif %}

    cdef _get_most_derived(self):
{% if derived %}
{% for m in derived %}
{% set dertype = m|d2cy %}
{% set cytype = m.replace('Ptr', '')|d2cy(False)|lower %}
        cdef {{m|d2cy}} tmp_ptr{{loop.index}} = \
        dynamicPointerPtrCast[{{m|d2cy(False)}}, {{class_type|flat(False)}}](self.c_ptr)
        cdef {{cytype}} res{{loop.index}}
        if tmp_ptr{{loop.index}}:
            res{{loop.index}} = {{cytype}}.__new__({{cytype}})
            res{{loop.index}}.c_ptr = tmp_ptr{{loop.index}}
            return res{{loop.index}}
{% endfor %}
{% endif %}
        return self

{% for m in get_methods %}
{% set frtype = m[0].replace('Ptr', '')|d2cy(False)|lower %}
    def {{m[1]|d2cy(False)}}(self, {{m[2]|d2cy(False)}} {{m[3]}}):
        ':type: {{frtype}} (or derived)'
        if not is_valid_ptrptr(self.c_ptr):
            raise ValueError("Pointer to oneDAL entity is NULL")
{% if 'Ptr' in m[0] %}
        cdef {{frtype}} res = {{frtype}}.__new__({{frtype}})
        res.c_ptr = get_{{flatname}}_{{m[1]}}(self.c_ptr, {{m[3]}})
{% if '_model' in frtype %}
        return res._get_most_derived()
{% else %}
        return res
{% endif %}
{% else %}
        return get_{{flatname}}_{{m[1]}}(self.c_ptr, {{m[3]}})
{% endif %}
{% endfor %}

    def __setstate__(self, state):
        if isinstance(state, bytes):
           self.c_ptr = deserialize_si[{{class_type|flat|strip(' *')}}](state)
        else:
           raise ValueError("Invalid state .....")

    def __getstate__(self):
        if self.c_ptr == NULL:
            raise ValueError("Pointer to oneDAL entity is NULL")
        bytes = serialize_si(self.c_ptr)
        return bytes

    def __reduce__(self):
        state_data = self.__getstate__()
        return (_rebuild, (self.__class__, state_data,))


cdef api void * unbox_{{flatname}}(a):
    return (<{{flatname}}>a).c_ptr


hpat_spec.append({
    'pyclass': {{flatname}},
    'c_name' : '{{flatname}}',
    'attrs'  : [
{% for m in enum_gets+named_gets %}
                ('{{m[1]}}', '{{m[1]|d2hpat(m[2], flatname) \
                if m in enum_gets else m[1]|d2hpat(m[0], flatname)}}'),
{% endfor %}
]})
{% else %}
%SNIP%
%SNIP%
{% endif %}
{% if not class_type.startswith('daal::'+ns) %}
cdef extern from "daal4py_cpp.h":
    ctypedef {{class_type|flat(False)}} {{(ns+'::'+splitname[-1])|flat|strip(' *')}}

{% set alias = (ns+'::'+splitname[-1]).replace('Ptr', '')|flat|lower|strip(' *') %}
{% set actual = flatname %}
{{alias}} = {{actual}}

hpat_spec.append({
    'c_name' : '{{alias}}',
    'alias' : '{{actual}}',
})
{% endif %}
"""

# macro generating C++ class for oneDAL interface classes
# accepts interface name and C++ type
gen_cpp_iface_macro = """
{% macro gen_cpp_iface(iface_name, iface_type) %}
class {{iface_name}}__iface__ : public {{parent|d2cy(False) \
                                if parent else 'algo_manager__iface__'}}
{
public:
    typedef {{iface_type}} daal_type;
    typedef {{iface_type}} {{iface_name}}Ptr_type;
    virtual {{iface_name}}Ptr_type get_ptr() = 0; //{return {{iface_name}}Ptr_type();}
};

typedef {{iface_name}}__iface__ c_{{iface_name}}__iface__;
typedef daal::services::SharedPtr<{{iface_name}}__iface__> c_{{iface_name}}__iface__sptr;

static {{iface_type}} to_daal(c_{{iface_name}}__iface__ * t) {
    return t ? t->get_ptr() : {{iface_type}}();
}
{% endmacro %}
"""

# macro generating cython class for oneDAL interface classes
# accepts interface name and C++ type
gen_cython_iface_macro = """
{% macro gen_cython_iface(iface_name, iface_type) %}
cdef extern from "daal4py_cpp.h":
    cdef cppclass c_{{iface_name}}__iface__\
    {{'(c_'+parent|d2cy(False)+')' if parent else ''}}:
        pass
    cdef cppclass c_{{iface_name}}__iface__sptr\
    {{'(c_'+parent|d2cy(False)+'sptr)' if parent else ''}}:
        c_{{iface_name}}__iface__sptr()
        c_{{iface_name}}__iface__sptr(c_{{iface_name}}__iface__*)
        c_{{iface_name}}__iface__sptr(c_{{iface_name}}__iface__sptr)
        c_{{iface_name}}__iface__ * get()
        void reset()


{% set inl = iface_name|lower + '__iface__' %}
{% if parent %}
cdef class {{inl}}({{parent|d2cy(False)|lower}}):
    pass
{% else %}
cdef class {{inl}}():
    cdef c_{{iface_name}}__iface__sptr c_ptr

    def __cinit__(self):
        self.c_ptr.reset()

    # we do not need __dealloc__(self), we store plain shared ptr
{% endif %}

hpat_spec.append({
    'pyclass'     : {{inl}},
    'c_name'      : '{{inl}}',
})
{% endmacro %}
"""

# macro generating typedefs in manager classes, e.g. algo and result types
# note that we will have multiple result types in dist-mode: each step has its own
gen_typedefs_macro = """
{% macro gen_typedefs(ns, template_decl, template_args,
                      mode="Batch", suffix="b", step_spec=None) %}
{% set disttarg = (step_spec.name.rsplit('__', 1)[0] + ', ') \
    if step_spec.name else "" %}
{% if template_decl|length > 0  %}
    typedef daal::{{ns}}::{{mode}}<\
    {{disttarg + template_args|fmt('{}', 'value')}}> \
    algo{{suffix}}_type;
{% else %}
    typedef daal::{{ns}}::{{mode}} algo{{suffix}}_type;
{% endif %}
{% if step_spec %}
    typedef {{step_spec.iomanager}}< \
    algo{{suffix}}_type, {{step_spec.output}}\
    {{(","+",".join(step_spec.iomargs)) if step_spec.iomargs else ""}} > \
    iom{{suffix}}_type;
{% else %}
{% if iombatch %}
    typedef {{iombatch}} iom{{suffix}}_type;
{% else %}
    typedef IOManager< \
    algo{{suffix}}_type, \
    daal::services::SharedPtr< typename algo{{suffix}}_type::ResultType > > \
    iom{{suffix}}_type;
{% endif %}
{% endif %}
{%- endmacro %}
"""

# macro for generate an algorithm instance with name algo$suffix
# This can also be used for steps of distributed mode
# set member=True if you want a init a member var (shared pointer)
gen_inst_algo = """
{% macro gen_inst(ns, params_req, params_opt, params_get,
                  create, suffix="", step_spec=None, member=False) %}
{% set algo = 'algo' + suffix %}
{% if step_spec.construct %}
{% set ctor = create + '(' + step_spec.construct + ')' %}
{% elif create  %}
{% set ctor = ('::create(_' + ', _'.join(create.keys()) + ')').replace('(_)', '()') %}
{% else %}
{% set ctor = '(' + params_req|fmt('to_daal({})', 'arg_member', sep=', ') + ')' %}
{% endif %}
{% if member %}
_{{algo}}{{' = (' if create else '.reset(new '}}{{algo}}_type{{ctor}});
{% elif step_spec.keepsstate %}
if(! _{{algo}}) _{{algo}}{{' = (' if create else '.reset(new '}}{{algo}}_type{{ctor}});
        {{algo}}_type * {{algo}} = _{{algo}}.get();
{% elif create %}
auto {{algo}} = {{algo}}_type{{ctor}};
{% else %}
auto {{algo}}_obj = {{algo}}_type{{ctor}};
        {{algo}}_type * {{algo}} = &{{algo}}_obj;
{% endif %}
{% if (step_spec == None or step_spec.params) and \
    params_get and params_opt|length and not create %}
        init_parameters({{('_' if member else '')+algo}}->{{params_get}});
{% else %}
        // skipping parameter initialization
{% endif %}
{%- endmacro %}
"""

# macro to generate the body of a compute function (batch and distributed)
gen_compute_macro = (
    gen_inst_algo
    + """
{% macro gen_compute(ns, input_args, params_req, params_opt, suffix="",
                     step_spec=None, tonative=True, iomtype=None, setupmode=False) %}
{% set iom = iomtype if iomtype else "iom"+suffix+"_type" %}
{% if step_spec %}

{% set input_len = step_spec.input|length %}
    (
{% if step_spec.addinput %}{% set add_offset = 0 %}
        {% for ia in step_spec.addinput %}\
        const std::vector< {{step_spec.input[add_offset+loop.index-1]}} > & \
        input{{add_offset+loop.index}}\
        {{'' if add_offset+loop.index==input_len else ', '}}\
        {% endfor %} //add inputs{% endif %}
{% if step_spec.setinput %}\
{% set set_offset = step_spec.addinput|length if step_spec.addinput else 0 %}

        {% for ia in step_spec.setinput %} \
        const {{step_spec.input[set_offset+loop.index-1]}} & \
        input{{set_offset+loop.index}}\
        {{'' if set_offset+loop.index==input_len==input_len else ', '}}\
        {% endfor %} //set inputs{% endif %}
{% if step_spec.dist_params %}

        {% for dp in step_spec.dist_params %}, \
        {{dp[0]}} {{dp[1]}}{% endfor %} //params{% endif %}
{% if step_spec.extrainput %}

        {{', ' + step_spec.extrainput \
        if step_spec.extrainput else ''}} //extra inputs{% endif %}

    )
    {
        {{gen_inst(ns, params_req, params_opt, params_get, create, suffix, step_spec)}}
{% for dp in step_spec.dist_params %}
        algo{{suffix}}->parameter.{{dp[1]}} = {{dp[1]}};
{% endfor %}

{% if step_spec.addinput %}

        int nr, i;
{% for ia in step_spec.addinput %}
        nr = 0, i = 0;
        for(auto data = input{{loop.index}}.begin();
            data != input{{loop.index}}.end();
            ++data, ++i) {
            if(*data) {
                algo{{suffix}}->input.add({{ia}}, to_daal(*data));
                ++nr;
            }
        }
        if(nr == 0 ) return typename {{iom}}::result_type();
{% endfor %}
{% endif %}

{% if step_spec.setinput %}
{% for ia in step_spec.setinput %}
        if(input{{set_offset+loop.index}})
            algo{{suffix}}->input.set({{ia}}, to_daal(input{{set_offset+loop.index}}));
{% endfor %}
{% endif %}
{% if step_spec.staticinput %}{% for ia in step_spec.staticinput %}
        if(! use_default(_{{ia[1]}}))
            algo{{suffix}}->input.set({{ia[0]}}, to_daal(_{{ia[1]}}));
{% endfor %}{% endif %}
{% else %}
{% if setupmode %}
(bool setup_only = false)
{% else %}
()
{% endif %}
    {
        ThreadAllow _allow_;
        auto algo{{suffix}} = _algo{{suffix}};

{% for ia in input_args %}
{% if "data_or_file" in ia.typ_cpp %}
        if(!{{ia.arg_member}}.table && {{ia.arg_member}}.file.size())
            {{ia.arg_member}}.table = readCSV({{ia.arg_member}}.file);
        if({{ia.arg_member}}.table)
            algo{{suffix}}->input.set({{ia.value}}, {{ia.arg_member}}.table);
{% else %}
        if({{ia.arg_member}})
            algo{{suffix}}->input.set({{ia.value}}, to_daal({{ia.arg_member}}));
{% endif %}
{% endfor %}
{% if setupmode %}

        if(setup_only) return typename iomb_type::result_type();
{% endif %}
{% endif %}

        algo{{suffix}}->compute();
{% if step_spec %}
        if({{iom}}::needsFini()) {
            algo{{suffix}}->finalizeCompute();
        }
{% endif %}
{% if tonative %}
        typename iomb_type::result_type daalres({{iom}}::getResult(*algo{{suffix}});
        int gc = 0;
{% if not step_spec %}        _allow.disallow();
{% endif %}
        NTYPE res = native_type(daalres, gc);
        TMGC(gc);
        return res;
{% else %}
        return {{iom}}::getResult(*algo{{suffix}});
{% endif %}
    }
{%- endmacro %}
"""
)

# generates the de-templetized *__iface__ struct with providing generic compute(...)
algo_iface_template = """
{% set prnt = iface[0]+'__iface__' if iface[0] else 'algo_manager__iface__' %}
struct {{algo}}__iface__ : public {{prnt}}
{
    {{distributed.decl_member}};
    {{streaming.decl_member}};
    {{algo}}__iface__({{[distributed, streaming]|fmt('{}', 'decl_dflt_cpp')}})
        : {{prnt}}()
          {{[distributed, streaming]|fmt(',{}', 'init_member', sep='\n')|indent(10) \
          if distributed.name or streaming.name else '\n'}}
    {}
{% set indent = 23+(result_map.class_type|length) %}
    virtual {{result_map.class_type}} * compute(
        {{input_args|fmt('{}', 'decl_cpp', sep=',\n')|indent(indent
    )}},
{{' '*indent}}bool setup_only = false)
        {assert(false); return NULL;}
{% if add_get_result %}
    virtual {{result_map.class_type}} * get_result() {assert(false); return NULL;}
{% endif %}
{% if streaming.name %}
    virtual {{result_map.class_type}} * finalize() {assert(false); return NULL;}
{% endif %}
};
"""

# generates "manager" class for managing distributed and batch modes of a given algo
manager_wrapper_template = (
    gen_typedefs_macro
    + gen_compute_macro
    + """
{% if template_decl|length == template_args|length %}
// The type used in cython
typedef {{algo}}__iface__  c_{{algo}}_manager__iface__;
typedef daal::services::SharedPtr<{{algo}}__iface__>  c_{{algo}}_manager__iface__sptr;

// The algo creation function
daal::services::SharedPtr<{{algo}}__iface__>
mk_{{algo}}({{params_all|fmt('{}', 'decl_cpp', sep=',\n')|indent(4+(algo|length))}});
{% endif %}

{% if template_decl  %}
template<{% for x in template_decl %}{{template_decl[x]['template_decl'] + \
' ' + x + ('' if loop.last else ', ')}}{% endfor %}>
{% endif %}
struct {{algo}}_manager{% if template_decl|length != template_args|length %}\
<{{template_args|fmt('{}', 'value')}}>{% endif %} : public {{algo}}__iface__
{
{{gen_typedefs(ns, template_decl, template_args, mode="Batch")}}
    {{args_all|fmt('{}', 'decl_member', sep=';\n')|indent(4)}};
    daal::services::SharedPtr< algob_type > _algob;

{% if streaming.name %}
{{gen_typedefs(ns, template_decl, template_args, mode="Online", suffix="stream")}}
    daal::services::SharedPtr< algostream_type > _algostream;
{% endif %}

    {{algo}}_manager({{params_ds|fmt(
        '{}', 'decl_cpp', sep=',\n'
    )|indent(13+algo|length)}})
        : {{algo}}__iface__({{[distributed, streaming]|fmt('{}', 'arg_cpp')}})
          {{args_all|fmt(',{}', 'init_member', sep='\n')|indent(10)}}
          , _algob()
{% if streaming.name %}
          , _algostream()
{% endif %}
    {
{% if streaming.name %}
      if({{streaming.arg_member}}) {
        {{gen_inst(ns, params_req, params_opt, params_get,
                   create, suffix="stream", member=True)}}
      } else
{% endif %}
      {
        {{gen_inst(ns, params_req, params_opt, params_get,
                   create, suffix="b", member=True)}}
      }
    }
    ~{{algo}}_manager()
    {
{% for i in args_decl %}
{% if 'data_or_file' in i %}
        delete _{{args_call[loop.index0]}};
{% elif '*' in i %}
        // ?? delete _{{args_call[loop.index0]}};
{% endif %}
{% endfor %}
    }

private:
{% for p in opt_params %}
{% if p[2]|length and not create %}
{% if p[1] and p[1][0][1]!='' %}
    template<{% for i in p[1] %}{{'typename' \
    if i[1]=='fptypes' else '::'.join(['daal', ns, i[1]])}} {{i[0]}}{% endfor %}>
{% endif %}
    void init_parameters(daal::{{p[0]}} & parameter)
    {
        {{p[2]|fmt('if(! use_default({})) parameter.{} = to_daal({});\
        ', 'arg_member', 'daalname', 'todaal_member', sep='\n')|indent(8)}}
    }
{% endif %}
{% endfor %}

{% for ifc in iface if ifc %}
    virtual {{ifc}}__iface__::{{ifc}}Ptr_type get_ptr()
    {
        return _algob;
    }
{% endfor %}

    typename iomb_type::result_type batch{{gen_compute(ns, input_args,
                                                       params_req, params_opt,
                                                       suffix="b",
                                                       iomtype=iombatch,
                                                       tonative=False, setupmode=True)}}

{% if streaming.name %}
    typename iomb_type::result_type stream{{gen_compute(ns, input_args,
                                                        params_req, params_opt,
                                                        suffix="stream",
                                                        iomtype=iombatch,
                                                        tonative=False)}}

    typename iomb_type::result_type * finalize()
    {
{% if distributed.name %}
        if({{distributed.arg_member}}) throw std::invalid_argument(
            "finalize() not supported in distributed mode"
        );
{% endif %}
        if({{streaming.arg_member}}) {
            _algostream->finalizeCompute();
            return new typename iomb_type::result_type(_algostream->getResult());
        } else {
            return new typename iomb_type::result_type(_algob->getResult());
        }
    }

{% endif %}

{% if step_specs is defined and distributed.name %}
    // Distributed computing
public:
{% for i in range(step_specs|length) %}
{{gen_typedefs(ns, template_decl, template_args, mode="Distributed",
               suffix=step_specs[i].name, step_spec=step_specs[i])}}
{% endfor %}

private:
{% for i in range(step_specs|length) if step_specs[i].keepsstate %}
    daal::services::SharedPtr< algo{{step_specs[i].name}}_type > \
    _algo{{step_specs[i].name}};
{% endfor %}

public:
{% for i in range(step_specs|length) %}
{% set sname = "run_"+step_specs[i].name %}
    typename iom{{step_specs[i].name}}_type::result_type {{sname + gen_compute(
        ns, input_args, params_req, params_opt, suffix=step_specs[i].name,
        step_spec=step_specs[i], tonative=False
    )}}

{% endfor %}
{% if enum_vals %}
{% for ev in enum_vals %}
    static const auto {{ev[0]}} = daal::{{ns}}::{{ev[1]}};
{% endfor %}
{% endif %}

{% set inp_names = step_specs[0].inputnames if (step_specs|length > 0) else inputnames %}
    static const int NI = {{inp_names|length}};

private:
    typename iomb_type::result_type distributed()
    {
{% for i in range(step_specs|length) if step_specs[i].keepsstate %}
        _algo{{step_specs[i].name}}.reset();
{% endfor %}
        return {{pattern}}::{{pattern}}< \
            {{algo}}_manager< {{template_args|fmt('{}', 'name')}} > \
            >::compute(*this, to_daal(_{{'), to_daal(_'.join(inp_names)}}));
    }
{% endif %}

public:
    typename iomb_type::result_type * compute(
        {{input_args|fmt('{}', 'decl_cpp', sep=',\n')|indent(46)}},
        bool setup_only = false)
    {
        {{input_args|fmt('{}', 'assign_member', sep=';\n')|indent(8)}};

{% set batchcall = '('+streaming.arg_member+' ? stream() : batch(setup_only))' \
    if streaming.name else 'batch(setup_only)'%}
{% if distributed.name %}
        typename iomb_type::result_type daalres = \
            {{distributed.arg_member + ' ? distributed() : ' + batchcall}};
        return new typename iomb_type::result_type(daalres);
{% else %}
        return new typename iomb_type::result_type({{batchcall}});
{% endif %}
    }

{% if add_get_result %}
    typename iomb_type::result_type * get_result()
    {
        return new typename iomb_type::result_type(iomb_type::getResult(*_algob));
    }
{% endif %}

};
"""
)

# generates cython class wrappers for given algo
# also generates defs for __iface__ class
parent_wrapper_template = """
cdef extern from "daal4py.h":
    # declare the C++ equivalent of the manager__iface__ class,
    # providing de-templatized access to compute
    cdef cppclass c_{{algo}}_manager__iface__\
    {{'(c_'+iface[0]+'__iface__)' if iface[0] else ''}}:
{% set indent = 17+(result_map.class_type|flat|length) %}
        {{result_map.class_type|flat}} compute(
            {{input_args|fmt('{}', 'decl_cyext', sep=',\n')|indent(indent)}},
{{' '*indent}}const bool setup_only) except +
{% if add_get_result %}
        {{result_map.class_type|flat}} get_result() except +
{% endif %}
{% if streaming.name %}
        {{result_map.class_type|flat}} finalize() except +
{% endif %}
    cdef cppclass c_{{algo}}_manager__iface__sptr \
    {{'(c_'+iface[0]+'__iface__sptr)' if iface[0] else ''}}:
        c_{{algo}}_manager__iface__sptr()
        c_{{algo}}_manager__iface__sptr(c_{{algo}}_manager__iface__*)
        c_{{algo}}_manager__iface__sptr(c_{{algo}}_manager__iface__sptr)
        c_{{algo}}_manager__iface__ * get()
        void reset()


cdef extern from "daal4py_cpp.h":
    # declare the C++ construction function.
    # Returns the manager__iface__ for access to de-templatized constructor
    cdef c_{{algo}}_manager__iface__sptr mk_{{algo}}(
        {{params_all|fmt('{}', 'decl_cyext', sep=',\n')|indent(37+2*(algo|length))}}
    ) except +

# this is our actual algorithm class for Python
cdef class {{algo}}{{'('+iface[0]|lower+'__iface__)' if iface[0] else ''}}:
    '''
    {{algo}}
    {{params_all|fmt('{}', 'sphinx', sep='\n')|indent(4)}}
    '''
    cdef tuple _params
    # Init simply forwards to the C++ construction function
    def __cinit__(self,
                  {{params_all|fmt('{}', 'decl_dflt_cy', sep=',\n')|indent(18)}}):
{% if distributed.name and streaming.name %}
        if {{[distributed, streaming]|fmt('{}', 'arg_py', sep=' and ')}}:
            raise ValueError('distributed streaming not supported')
{% endif %}
        self.c_ptr = mk_{{algo}}(
            {{params_all|fmt('{}', 'arg_cyext', sep=',\n')|indent(25+(algo|length))}}
        )
        current_locals = locals()
        ordered_input_args = '''
            {{params_all|fmt('{}', 'name', sep=' ')|indent(0)}}
        '''.strip().split()
        self._params = tuple(
            current_locals[arg]
            for arg in ordered_input_args
        )

    def __reduce__(self):
        return (self.__class__, self._params)

{% if not iface[0] %}
    # the C++ manager__iface__ (de-templatized)
    cdef c_{{algo}}_manager__iface__sptr c_ptr

    # we do not need __dealloc__(self), we store plain shared ptr
{% endif %}

{% set cytype = result_map.class_type.replace('Ptr', '')|d2cy(False)|lower %}
    # compute simply forwards to the C++ de-templatized manager__iface__::compute
    def _compute(self,
                 {{input_args|fmt('{}', 'decl_dflt_cy', sep=',\n')|indent(17)}},
                 setup=False):
        if self.c_ptr.get() == NULL:
            raise ValueError("Pointer to oneDAL entity is NULL")
        {{input_args|fmt('{}', 'get_obj', sep='\n')|indent(8)}}
        cdef c_{{algo}}_manager__iface__* algo = \
            <c_{{algo}}_manager__iface__*>self.c_ptr.get()
        # we cannot have a constructor accepting a c-pointer,
        # so we split into construction and setting pointer
        cdef {{cytype}} res = {{cytype}}.__new__({{cytype}})
        res.c_ptr = deref(algo).compute(
            {{input_args|fmt('{}', 'arg_cyext', sep=',\n')|indent(40)}}, setup)
        return res

    # compute simply forwards to the C++ de-templatized manager__iface__::compute
    def compute(self,
                {{input_args|fmt('{}', 'decl_dflt_cy', sep=',\n')|indent(16)}},
                setup=False):
        '''
        {{algo}}.compute({{input_args|fmt('{}', 'name', sep=', ')}})
        Do the actual computation on provided input data.

        {{input_args|fmt('{}', 'sphinx', sep='\n')|indent(8)}}
        :rtype: {{cytype}}
        '''
        return self._compute({{input_args|fmt('{}', 'name', sep=', ')}}, False)

{% if add_get_result %}
    def __get_result__(self):
        if self.c_ptr.get() == NULL:
            raise ValueError("Pointer to oneDAL entity is NULL")
        cdef c_{{algo}}_manager__iface__* algo = \
            <c_{{algo}}_manager__iface__*>self.c_ptr.get()
        # we cannot have a constructor accepting a c-pointer,
        # so we split into construction and setting pointer
        cdef {{cytype}} res = {{cytype}}.__new__({{cytype}})
        res.c_ptr = deref(algo).get_result()
        return res
{% endif %}

{% if streaming.name %}
    # finalize simply forwards to the C++ de-templatized manager__iface__::finalize
    def finalize(self):
        if self.c_ptr.get() == NULL:
            raise ValueError("Pointer to oneDAL entity is NULL")
        cdef c_{{algo}}_manager__iface__* algo = \
            <c_{{algo}}_manager__iface__*>self.c_ptr.get()
        # we cannot have a constructor accepting a c-pointer,
        # so we split into construction and setting pointer
        cdef {{cytype}} res = {{cytype}}.__new__({{cytype}})
        res.c_ptr = deref(algo).finalize()
        return res
{% endif %}

{% if add_setup %}
    # setup forwards to the C++ de-templatized
    # manager__iface__::compute(..., setup_only=true)
{% set setup_required = [] %}
{% set setup_optional = [] %}
{% for ia in input_args %}
{% if ia.name in add_setup %}
{% set setup_required = setup_required.append(ia) %}
{% else %}
{% set setup_optional = setup_optional.append(ia) %}
{% endif %}
{% endfor %}
    def setup(self,
              {{setup_required|fmt('{}', 'decl_cy', sep=',\n')|indent(14)}},
              {{setup_optional|fmt('{} = None', 'decl_cy', sep=',\n')|indent(14)}}):
        '''
        {{algo}}.setup({{input_args|fmt('{}', 'name', sep=', ')}})
        Setup (partial) input data for using algorithm object in other algorithms.

        {{input_args|fmt('{}', 'sphinx', sep='\n')|indent(8)}}
        :rtype: None
        '''
        return self._compute({{input_args|fmt('{}', 'name', sep=', ')}}, True)
{% endif %}

{% if streaming.name %}
    # finalize simply forwards to the C++ de-templatized manager__iface__::finalize
    def finalize(self):
        if self.c_ptr.get() == NULL:
            raise ValueError("Pointer to oneDAL entity is NULL")
        cdef c_{{algo}}_manager__iface__* algo = \
            <c_{{algo}}_manager__iface__*>self.c_ptr.get()
        # we cannot have a constructor accepting a c-pointer,
        # so we split into construction and setting pointer
        cdef {{cytype}} res = {{cytype}}.__new__({{cytype}})
        res.c_ptr = deref(algo).finalize()
        return res
{% endif %}

cdef api void * unbox_{{algo}}(object a):
    algo = <{{algo}}>a
    return _daal_clone(algo.c_ptr)
"""

# generates the C++ algorithm construction function
# it all it does is dispatching to the template managers from given arguments
# tfactory is a recursive jinja2 macro to handle any number of template args
algo_wrapper_template = """
{% macro tfactory(tmpl_spec, prefix, params, dist=False, args=[], indent=4) %}
{% for a in tmpl_spec[0][1]['values'] %}
{{" "*indent}}if({{tmpl_spec[0][0]}} == "{{a.rsplit('::',1)[-1]}}") {
{% if tmpl_spec|length == 1 %}
{% set algo_type = prefix + '<' + ', '.join(args+[a]) + ' >' %}
{{" "*(indent+4)}}return daal::services::SharedPtr<{{algo}}__iface__>(
    new {{algo_type}}({{params|fmt('{}', 'arg_cpp')}})
);
{% else %}
{{tfactory(tmpl_spec[1:], prefix, params, dist, args+[a], indent+4)}}
{% endif %}
{{" "*(indent)}}} else {% if loop.last %} {
{{" "*(indent+4)}} throw std::runtime_error(
std::string(
"Error in {{algo}}: Cannot handle unknown value for parameter '{{tmpl_spec[0][0]}}': "
) + {{tmpl_spec[0][0]}} + "'"
);
{{" "*(indent)}}}
{% endif %}
{% endfor %}
{%- endmacro %}

daal::services::SharedPtr<{{algo}}__iface__>
mk_{{algo}}({{params_all|fmt('{}', 'decl_cpp', sep=',\n')|indent(4+(algo|length))}})
{
    ThreadAllow _allow_;
{% if template_decl %}
{{tfactory(template_decl.items()|list, algo+'_manager', params_ds, dist=dist)}}
    throw std::runtime_error("Error: Could not construct {{algo}}.");
    return daal::services::SharedPtr<{{algo}}__iface__>();
{% else %}
    return daal::services::SharedPtr<{{algo}}__iface__>(new {{algo}}_manager(
        {{params_ds|fmt('{}', 'arg_cpp')}})
    );
{% endif %}
}

// A C interface to compute, used for HPAT
extern "C" void * compute_{{algo}}(daal::services::SharedPtr<{{algo}}__iface__> * algo,
{{' '*(27+(algo|length))}}{{input_args|fmt(
    '{}', 'decl_c', sep=',\n')|indent(27+(algo|length))}},
{{' '*(27+(algo|length))}}bool setup=false)
{
{% if distributed.name %}
    (*algo)->{{distributed.arg_member}} = c_num_procs() > 0;
{% endif %}
    void * res = (*algo)->compute({{input_args|fmt('{}', 'arg_c', sep=',\n')|indent(34)}},
                                  setup);
    return res;
};
"""

# generate a D4PSpec
hpat_spec_template = """
hpat_spec.append({
    'pyclass'     : {{algo}},
    'c_name'      : '{{algo}}',
    'params'      : [{{params_all|fmt('{}', 'spec', sep=',\n')|indent(21)}}],
    'input_types' : [{{input_args|fmt('{}', 'spec', sep=',\n')|indent(21)}}],
    'result_dist' : {{"'REP'" if step_specs is defined else "None"}},
    'has_setup'   : {{True if add_setup else False}}
})
"""

# template for footer in .pyx file
# requires {{algos}}    list of algorithms that are available
#          {{version}}  version of DAAL
pyx_footer_template = """
def getTreeState(model, i=0, n_classes=1):
    cdef TreeState cTreeState
    if False:
        pass
{% for model in ['algorithms::decision_forest::classification',
                 'algorithms::gbt::classification'] %}
{% if model in algos %}
{% set flatname = '::'.join([model, 'model'])|flat %}
    elif isinstance(model, {{flatname}}):
        cTreeState = _getTreeState((<{{flatname}}>model).c_ptr, i, n_classes)
{% endif %}
{% endfor %}
{% for model in ['algorithms::decision_tree::classification'] %}
{% if model in algos %}
{% set flatname = '::'.join([model, 'model'])|flat %}
    elif isinstance(model, {{flatname}}):
        cTreeState = _getTreeState((<{{flatname}}>model).c_ptr, n_classes)
{% endif %}
{% endfor %}
{% for model in ['algorithms::decision_forest::regression',
                 'algorithms::gbt::regression'] %}
{% if model in algos %}
{% set flatname = '::'.join([model, 'model'])|flat %}
    elif isinstance(model, {{flatname}}):
        cTreeState = _getTreeState((<{{flatname}}>model).c_ptr, i, 1)
{% endif %}
{% endfor %}
{% for model in ['algorithms::decision_tree::regression'] %}
{% if model in algos %}
{% set flatname = '::'.join([model, 'model'])|flat %}
    elif isinstance(model, {{flatname}}):
        cTreeState = _getTreeState((<{{flatname}}>model).c_ptr, 1)
{% endif %}
{% endfor %}
    else:
        assert(False), 'Incorrect model type: ' + str(type(model))
    state = pyTreeState()
    state.set(&cTreeState)
    return state


cdef extern from "daal4py_version.h":
    cdef const long long INTEL_DAAL_VERSION
    cdef const long long __INTEL_DAAL_BUILD_DATE
    cdef const std_string __INTEL_DAAL_STATUS__
    cppclass LibraryVersionInfo:
        LibraryVersionInfo()
        int majorVersion, minorVersion, updateVersion
        char * build_rev
def _get__version__():
    return "{}".format({{version}})
def _get__daal_link_version__():
    return "{}{}_{}".format(INTEL_DAAL_VERSION, __INTEL_DAAL_STATUS__,
                            __INTEL_DAAL_BUILD_DATE)
def _get__daal_run_version__():
    cdef LibraryVersionInfo li
    return "{}{}{}_{}".format(li.majorVersion, str(li.minorVersion).zfill(2),
                              str(li.updateVersion).zfill(2), li.build_rev)

"""

# template for footer in .cpp file
# requires {{algos}}                list of algorithms that are available
#          {{dist_custom_algos}}    list of algorithms with custom distribution pattern
cpp_footer_template = """
{% for algo in dist_custom_algos%}
{% if algo in algos %}
#include "dist_{{algo|flat}}.h"
{% endif %}
{% endfor %}
"""


##################################################################################
# A set of jinja2 filters to convert arguments, types etc which where extracted
# from oneDAL C++ headers to cython syntax and/or C++ for our own code
##################################################################################
def flat(t, cpp=True):
    """Flatten C++ name, leaving only what's needed to disambiguate names.
    E.g. stripping of leading namespaces and replaceing :: with _
    """

    def _flat(ty):
        def __flat(typ):
            nn = typ.split("::")
            if nn[0] == "daal":
                if nn[1] == "algorithms":
                    r = "_".join(nn[2:])
                else:
                    r = "_".join(nn[1:])
            elif nn[0] == "algorithms":
                r = "_".join(nn[1:])
            else:
                r = "_".join(nn)
            return (
                ("c_" if cpp and typ.endswith("__iface__") else "")
                + r
                + (
                    " *"
                    if cpp and any(typ.endswith(x) for x in ["__iface__", "Ptr"])
                    else ""
                )
            )

        ty = ty.replace(
            "daal::algorithms::kernel_function::KernelIfacePtr",
            "daal::services::SharedPtr<kernel_function::KernelIface>",
        )
        ty = re.sub(
            r"(daal::)?(algorithms::)?(engines::)?EnginePtr",
            r"daal::services::SharedPtr<engines::BatchBase>",
            ty,
        )
        ty = re.sub(
            r"(?:daal::)?(?:algorithms::)?([^:]+::)BatchPtr",
            r"daal::services::SharedPtr<\1Batch>",
            ty,
        )
        ty = re.sub(r"(daal::)?services::SharedPtr<([^>]+)>", r"\2__iface__", ty)
        return " ".join([__flat(x).replace("const", "") for x in ty.split(" ")])

    return [_flat(x) for x in t if x] if isinstance(t, list) else _flat(t)


def d2cy(ty, cpp=True):
    def flt(t, cpp):
        return flat(t, cpp).replace("lambda", "lambda_")

    return [flt(x, cpp) for x in ty if x] if isinstance(ty, list) else flt(ty, cpp)


def d2hpat(arg, ty, fn):
    def flt(arg, t):
        rtype = d2cy(t)
        if fn in hpat_types and arg in hpat_types[fn]:
            return hpat_types[fn][arg]
        return (
            "dtable_type"
            if "NumericTablePtr" in rtype
            else rtype.replace("ModelPtr", "model").replace(" ", "")
        )

    return [flt(x, y) for x, y in zip(arg, ty)] if isinstance(ty, list) else flt(arg, ty)


def fmt(*args, **kwargs):
    sep = kwargs["sep"] if "sep" in kwargs else ", "
    return sep.join([y for y in [x.format(args[1], *args[2:]) for x in args[0]] if y])


jenv = jinja2.Environment(
    trim_blocks=True,
    autoescape=select_autoescape(disabled_extensions=("pyx"), default_for_string=False),
)
# jenv.filters['match'] = lambda a, x: [x for x in a if s in x]
jenv.filters["d2cy"] = d2cy
jenv.filters["flat"] = flat
jenv.filters["d2hpat"] = d2hpat
jenv.filters["strip"] = lambda s, c: s.strip(c)
jenv.filters["quote"] = lambda x: "'" + x + "'" if x else ""
jenv.filters["fmt"] = fmt


class wrapper_gen(object):
    def __init__(self, ac, ifaces):
        self.algocfg = ac
        self.ifaces = ifaces

    def gen_headers(self):
        """
        return code for initing
        """
        cpp = (
            "#ifndef DAAL4PY_CPP_INC_\n"
            + "#define DAAL4PY_CPP_INC_\n#include <daal4py_dist.h>\n\n"
        )
        pyx = ""
        for i in self.ifaces:
            tstr = (
                gen_cython_iface_macro
                + '{{gen_cython_iface("'
                + i
                + '", "'
                + self.ifaces[i][0]
                + '")}}\n'
            )
            t = jenv.from_string(tstr)
            pyx += t.render({"parent": self.ifaces[i][1]}) + "\n"
            tstr = (
                gen_cpp_iface_macro
                + '{{gen_cpp_iface("'
                + i
                + '", "'
                + self.ifaces[i][0]
                + '")}}\n'
            )
            t = jenv.from_string(tstr)
            cpp += t.render({"parent": self.ifaces[i][1]}) + "\n"

        return (cpp, cython_header + pyx)

    ##################################################################################
    def gen_modelmaps(self, ns, algo):
        """
        return string from typemap_wrapper_template for given Model.
        uses entries from 'gets' in Model class def to fill 'named_gets'.
        """
        jparams = self.algocfg[ns + "::" + algo]["model_typemap"]
        if len(jparams) > 0:
            jparams["ns"] = ns
            jparams["algo"] = algo
            t = jenv.from_string(typemap_wrapper_template)
            return (t.render(**jparams) + "\n").split("%SNIP%")
        return "", "", ""

    ##################################################################################
    def gen_resultmaps(self, ns, algo):
        """
        Generates typedefs for Result type of given namespace.
        Uses target language-specific defines/functions
          - native_type: returns native representation of its argument
          - TMGC(n): deals with GC(refcounting for given number of references (R)
          -
        Looks up Return type and then target-language
        independently creates lists of its content.
        """
        jparams = self.algocfg[ns + "::" + algo]["result_typemap"]
        if len(jparams) > 0:
            jparams["ns"] = ns
            jparams["algo"] = algo
            t = jenv.from_string(typemap_wrapper_template)
            return (t.render(**jparams) + "\n").split("%SNIP%")
        return "", "", ""

    def lp(self, t):
        tmp = t.split("\n")
        for i in enumerate(tmp):
            print(i[0], i[1])

    ##################################################################################
    def gen_wrapper(self, ns, algo):
        """
        Here we actually generate the wrapper code. Separating this from preparation
        allows us to cross-reference between algos, for example for multi-phased algos.

        We combine the argument (template, input, parameter) information appropriately.
        We take care of the right order and bring them
        in the right format for our jinja templates.
        We pass them to the templates in a dict jparams, used a globals vars for jinja.

        Handling single-phased algos only which are not part of a multi-phased algo
        """
        cfg = self.algocfg[ns + "::" + algo]
        cpp_begin, pyx_begin, pyx_end, typesstr = "", "", "", ""

        cpp_map, cpp_end, pyx_map = self.gen_modelmaps(ns, algo)
        a, b, c = self.gen_resultmaps(ns, algo)
        cpp_map += a
        cpp_end += b
        pyx_map += c

        if len(cfg["params"]) == 0:
            return (cpp_map, cpp_begin, cpp_end, pyx_map, pyx_begin, pyx_end, typesstr)

        jparams = cfg["params"].copy()
        jparams.update(cfg["params"]["params_templ"])
        jparams["create"] = cfg["create"]
        jparams["add_setup"] = cfg["add_setup"]
        jparams["add_get_result"] = cfg["add_get_result"]
        jparams["model_maps"] = cfg["model_typemap"]
        jparams["result_map"] = cfg["result_typemap"]
        jparams["params_ds"] = (
            jparams["params_req"]
            + jparams["params_opt"]
            + [cfg["distributed"], cfg["streaming"]]
        )
        jparams["params_all"] = (
            jparams["params_req"]
            + (jparams["template_args"] if jparams["template_args"] else [])
            + jparams["params_opt"]
            + [cfg["distributed"], cfg["streaming"]]
        )
        jparams["args_all"] = (
            jparams["input_args"] + jparams["params_req"] + jparams["params_opt"]
        )

        for p in ["distributed", "streaming"]:
            if p in cfg:
                jparams[p] = cfg[p]

        t = jenv.from_string(algo_iface_template)
        cpp_begin += t.render(**jparams) + "\n"

        if jparams:
            if "dist" in cfg:
                # a wrapper for distributed mode
                jparams.update(cfg["dist"])

            t = jenv.from_string(manager_wrapper_template)
            cpp_begin += t.render(**jparams) + "\n"

            t = jenv.from_string(hpat_spec_template)
            pyx_begin += t.render(**jparams) + "\n"

            # this is our actual API wrapper
            t = jenv.from_string(parent_wrapper_template)
            pyx_end += t.render(**jparams) + "\n"

            # the C function generating specialized classes
            t = jenv.from_string(algo_wrapper_template)
            cpp_end += t.render(**jparams) + "\n"

        return (cpp_map, cpp_begin, cpp_end, pyx_map, pyx_begin, pyx_end, typesstr)

    ##################################################################################
    def gen_footers(
        self, no_dist=False, no_stream=False, algos=[], version="", dist_custom_algos=[]
    ):
        t = jenv.from_string(pyx_footer_template)
        pyx_footer = t.render(algos=algos, version=version)
        pyx_footer += "__has_dist__ = {}\n\n".format(not no_dist)

        if no_dist:
            return ("", pyx_footer, "")
        t = jenv.from_string(cpp_footer_template)
        cpp_footer = t.render(algos=algos, dist_custom_algos=dist_custom_algos)
        return ("", pyx_footer, cpp_footer)
