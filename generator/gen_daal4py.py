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
# Top level code for generating code for building daal4py
# - Uses parse.py to read in oneDAL C++ headers files.
# - Extracts necessary information like enum values, namespaces, templates etc
# - Uses wrapper_gen.py to generate code
#   - C++ code to construct objects and call algorithms (shared and distributed memory)
#   - Cython code to generate python API
###############################################################################

import os
import re
import shutil
from collections import OrderedDict, defaultdict
from os.path import isdir
from os.path import join as jp
from shutil import copytree, rmtree
from subprocess import call
from sys import platform

from .format import mk_var
from .parse import parse_header, parse_version
from .wrapper_gen import wrapper_gen
from .wrappers import (
    add_get_result,
    add_setup,
    defaults,
    enum_maps,
    enum_params,
    has_dist,
    ifaces,
    ignore,
    no_constructor,
    no_warn,
    required,
    result_to_compute,
    wrap_algo,
)

try:
    basestring
except NameError:
    basestring = str


def cpp2hl(cls):
    return cls.replace("::", "_")


###############################################################################
def cleanup_ns(fname, ns):
    """return a sanitized namespace name"""
    # strip of namespace 'interface1'
    while len(ns) != 0 and ns[-1].startswith("interface"):
        del ns[-1]
    # cleanup duplicates
    while len(ns) >= 2 and ns[-1] == ns[len(ns) - 2]:
        del ns[-1]
    # we should now have our namespace hierachy
    if len(ns) == 0 or ns[0] != "daal":
        print(
            fname + ":0: Warning: No namespace (starting with daal) found in"
            " " + fname + ". Ignored."
        )
        return False
    nsn = "::".join(ns[1:])
    # namespace 'daal' is special, it's empty
    if len(nsn) == 0:
        nsn = "daal"
    # Multiple namespaces will leave 'interface' in the hierachy
    # we cannot handle these cases
    if "interface" in nsn:
        print(fname + ":0: Warning: Multiple namespaces found in " + fname + ". Ignored.")
        return False
    return nsn


###############################################################################
###############################################################################
def splitns(x):
    """Split string at last '::'"""
    tmp_ = x.rsplit("::", 1)
    if len(tmp_) == 1:
        return ("", x)
    return tmp_


def get_parent(ns):
    tmp = ns.rsplit("::", 1)
    return tmp[0] if len(tmp) > 1 else "daal"


###############################################################################
class namespace(object):
    """Holds all extracted data of a namespace"""

    def __init__(self, name):
        self.classes = {}
        self.enums = {}
        self.typedefs = {}
        self.headers = []
        self.includes = set()
        self.name = name
        self.need_methods = False
        self.steps = set()
        self.children = set()


###############################################################################
def ignored(ns, a=None):
    return ns in ignore and (
        (a is not None and a in ignore[ns]) or (a is None and not ignore[ns])
    )


###############################################################################
###############################################################################
class cython_interface(object):
    """collecting and generating data for code generation"""

    # classes/functions we generally ignore
    ignores = [
        "AlgorithmContainerIface",
        "AnalysisContainerIface",
        "PredictionContainerIface",
        "TrainingContainerIface",
        "DistributedPredictionContainerIface",
        "BatchContainerIface",
        "OnlineContainerIface",
        "DistributedContainerIface",
        "BatchContainer",
        "OnlineContainer",
        "DistributedContainer",
        "serializeImpl",
        "deserializeImpl",
        "serialImpl",
        "getEpsilonVal",
        "getMinVal",
        "getMaxVal",
        "getPMMLNumType",
        "getInternalNumType",
        "getIndexNumType",
        "allocateNumericTableImpl",
        "allocateImpl",
        "allocate",
        "initialize",
        "setPartialResultStorage",
        "addPartialResultStorage",
    ]

    # files we ignore/skip
    ignore_files = [
        "daal_shared_ptr.h",
        "daal.h",
        "daal_sycl.h",
        "daal_win.h",
        "algorithm_base_mode_batch.h",
        "algorithm_base.h",
        "algorithm.h",
        "ridge_regression_types.h",
        "kdtree_knn_classification_types.h",
        "multinomial_naive_bayes_types.h",
        "daal_kernel_defines.h",
        "linear_regression_types.h",
    ]

    done = []

    ###############################################################################
    def __init__(self, include_root):
        self.include_root = include_root
        self.namespace_dict = defaultdict(namespace)

    ###############################################################################
    def read(self):
        """
        Walk through each directory in the root dir and read in C++ headers.
        Creating a namespace dictionary.
        Of course, it needs to go through every header file to find out
        what namespace it is affiliated with.
        Once done, we have a dictionary where the key is the namespace
        and the values are namespace class objects.
        These objects carry all information as extracted by parse.py.
        """
        print("reading headers from " + self.include_root)
        for dirpath, dirnames, filenames in os.walk(self.include_root):
            for filename in filenames:
                if (
                    filename.endswith(".h")
                    and "neural_networks" not in dirpath
                    and not any(
                        filename.endswith(x) for x in cython_interface.ignore_files
                    )
                ):
                    fname = jp(dirpath, filename)
                    with open(fname, "r") as header:
                        parsed_data = parse_header(header, cython_interface.ignores)
                    ns = cleanup_ns(fname, parsed_data["ns"])
                    # Now let's update the namespace;
                    # more than one file might contribute to the same ns
                    if ns:
                        if ns not in self.namespace_dict:
                            self.namespace_dict[ns] = namespace(ns)
                        pns = get_parent(ns)
                        if pns not in self.namespace_dict:
                            self.namespace_dict[pns] = namespace(pns)
                        if ns != "daal":
                            self.namespace_dict[pns].children.add(ns)
                        self.namespace_dict[ns].includes = self.namespace_dict[
                            ns
                        ].includes.union(parsed_data["includes"])
                        self.namespace_dict[ns].steps = self.namespace_dict[
                            ns
                        ].steps.union(parsed_data["steps"])
                        # we support multiple interface* namespaces for class defs
                        for c in parsed_data["classes"]:
                            if "interface" not in c:
                                self.namespace_dict[ns].classes[c] = parsed_data[
                                    "classes"
                                ][c]
                            else:
                                tmp = splitns(c)
                                subns = "{}::{}".format(ns, tmp[0])
                                if subns not in self.namespace_dict:
                                    self.namespace_dict[subns] = namespace(subns)
                                self.namespace_dict[subns].classes[tmp[1]] = parsed_data[
                                    "classes"
                                ][c]
                                self.namespace_dict[subns].classes[tmp[1]].name = tmp[1]

                        self.namespace_dict[ns].enums.update(parsed_data["enums"])
                        self.namespace_dict[ns].typedefs.update(parsed_data["typedefs"])
                        self.namespace_dict[ns].headers.append(
                            fname.replace(self.include_root, "").lstrip("/")
                        )
                        if parsed_data["need_methods"]:
                            self.namespace_dict[ns].need_methods = True
        with open(
            jp(self.include_root, "..", "services", "library_version_info.h")
        ) as header:
            v = parse_version(header)
            self.version = (int(v[0]), int(v[1]), int(v[2]), str(v[3]))
            print("Found oneDAL version {}.{}.{}.{}".format(*self.version))

    ###############################################################################
    # Postprocessing starts here
    ###############################################################################
    def get_ns(self, ns, c_, attrs=["classes", "enums", "typedefs"]):
        """
        Find class c starting in namespace ns.
        We search all entries given in attrs - not only 'classes'.
        c can have qualified namespaces in its name.
        We go up to all enclosing namespaces of ns
        until we find c (e.g. current-namespace::c)
        or we reached the global namespace.
        """
        if not c_.startswith("interface"):
            # Let's get rid of 'interface*'
            c = re.sub(r"interface\d+::", r"", c_)
        else:
            c = c_
        # we need to cut off leading daal::
        if c.startswith("daal::"):
            c = c[6:]
        tmp = splitns(c)
        cns = "::" + tmp[0]  # the namespace-part of our class
        cname = tmp[-1]  # class name (stripped off namespace)
        currns = ns + cns  # current namespace in which we look for c
        done = False
        while currns and not done:
            # if in the outmost level we only have cns, which starts with '::'
            tmpns = currns.strip(":")
            if tmpns in self.namespace_dict and any(
                cname in getattr(self.namespace_dict[tmpns], a) for a in attrs
            ):
                return tmpns
            if ns == "daal":
                done = True
            else:
                if currns.startswith("::"):
                    ns = "daal"
                else:
                    ns = splitns(ns)[0]
                currns = ns + cns
        return None

    ###############################################################################
    def get_all_attrs(self, ns, cls, attr, ons=None):
        """
        Return an ordered dict, combining the 'attr' dicts of
        class 'cls' and all its parents.
        Note: this looks for parents of 'cls' not parents of 'ns'!
        """
        if not ons:
            ons = ns
        # we need to cut off leading daal::
        cls = cls.replace("daal::", "")
        if ns not in self.namespace_dict or cls not in self.namespace_dict[ns].classes:
            if ns in self.namespace_dict and "::" not in cls:
                # the class might be in one of the parent ns
                ns = self.get_ns(ns, cls)
                if ns is None:
                    return None
            else:
                return None

        pmembers = OrderedDict()
        # we add ours first. When expanding, duplicates from parents will be ignored.
        tmp = getattr(self.namespace_dict[ns].classes[cls], attr)
        for a in tmp:
            n = a if "::" in a else ns + "::" + a
            if not ignored(ons, n):
                pmembers[n] = tmp[a]
        for parent in self.namespace_dict[ns].classes[cls].parent:
            # we need to cut off leading daal::
            sanep = parent.split()[-1].replace("daal::", "")
            parentclass = splitns(sanep)[1]
            pns = self.get_ns(ns, sanep)
            if (
                pns is not None
                and "interface" not in parent
                and ns == pns
                and self.namespace_dict[ns].classes[cls].iface
                != self.namespace_dict[pns].classes[parentclass].iface
            ):
                sanep = "{}::{}".format(self.namespace_dict[ns].classes[cls].iface, sanep)
                pns = self.get_ns(ns, sanep)
            if pns is not None:
                pms = self.get_all_attrs(pns, parentclass, attr, ons)
                for x in pms:
                    # ignore duplicates from parents
                    if not ignored(ons, x) and not any(x == y for y in pmembers):
                        pmembers[x] = pms[x]
        return pmembers

    ###############################################################################
    def to_lltype(self, p, t):
        """
        return low level (C++ type). Usually the same as input.
         Only very specific cases need a conversion.
        """
        if p in enum_params:
            return enum_params[p]
        if t in ["DAAL_UINT64"]:
            return "ResultToComputeId"
        return t

    ###############################################################################
    def to_hltype(self, ns, t):
        """
        Return triplet (type, {'stdtype'|'enum'|'class'|'?'}, namespace)
        to be used in the interface for given type 't'.
            'stdtype' means 't' is a standard data type understood by
            cython and plain C++
            'enum' means 't' is a C/C++ enumeration
            'class' means 't' is a regular C++ class
            '?' means we do not know what 't' is
        For classes, we also add lookups in namespaces that
        oneDAL C++ API finds through "using".
        """
        tns, tname = splitns(t)
        if t in [
            "double",
            "float",
            "int",
            "size_t",
        ]:
            return (t, "stdtype", "")
        if t in ["bool"]:
            return ("bool", "stdtype", "")
        if t == "algorithmFPType":
            return ("double", "stdtype", "")
        if t.endswith("ModelPtr"):
            thens = self.get_ns(ns, t, attrs=["typedefs"])
            return ("daal::" + thens + "::ModelPtr", "class", tns)
        if t.endswith("ResultPtr"):
            thens = self.get_ns(ns, t, attrs=["typedefs"])
            return ("daal::" + thens + "::ResultPtr", "class", tns)
        if t in ["data_management::NumericTablePtr"] or any(
            t == x[0] for x in ifaces.values()
        ):
            return ("daal::" + t, "class", tns)
        if t.endswith("KeyValueDataCollectionPtr"):
            return ("dict_NumericTablePtr", "class", "")
        if t.endswith("DataCollectionPtr"):
            return ("list_NumericTablePtr", "class", "")
        if (
            "Batch" in self.namespace_dict[ns].classes
            and t in self.namespace_dict[ns].classes["Batch"].typedefs
        ):
            tns, tname = splitns(self.namespace_dict[ns].classes["Batch"].typedefs[t])
            return (self.namespace_dict[ns].classes["Batch"].typedefs[t], "class", tns)
        if "services::SharedPtr" in t:
            no_spt = re.sub(r".*services::SharedPtr<(.*)>.*", r"\1", t).strip()
            no_spt_ns = self.get_ns(ns, no_spt)
            tt = "daal::services::SharedPtr<daal::{}::{}>".format(
                no_spt_ns, splitns(no_spt)[1]
            )
        else:
            tt = re.sub(r"(?<!daal::)algorithms::", r"daal::algorithms::", t)
        if any(tt == x[0] for x in ifaces.values()):
            return (tt, "class", tns)
        tns = self.get_ns(ns, t)
        if tns:
            tt = tns + "::" + tname
            if tt == t:
                return (
                    ("std::string &", "enum", tns)
                    if tname in self.namespace_dict[tns].enums
                    else (tname, "class", tns)
                )
            return self.to_hltype(ns, tt)
        usings = ["algorithms::optimization_solver"]
        if not any(t.startswith(x) for x in usings):
            for nsx in usings:
                r = self.to_hltype(ns, nsx + "::" + t)
                if r:
                    return r
        return None if "::" in t else (t, "??", "??")

    ###############################################################################
    def get_values(self, ns, n):
        """
        Return available values for given enum or special "group".
        """
        if n == "fptypes":
            return ["double", "float"]
        nns = self.get_ns(ns, n)
        if nns:
            nn = splitns(n)[1]
            if nn in self.namespace_dict[nns].enums:
                return [
                    re.sub(
                        r"(?<!daal::)algorithms::", r"daal::algorithms::", nns + "::" + x
                    )
                    for x in self.namespace_dict[nns].enums[nn]
                ]
            return ["unknown_" + nns + "_class_" + n]
        return ["unknown_" + n]

    ###############################################################################
    def get_tmplarg(self, ns, n):
        """
        Return template argument specifier: "typename" or actual type.
        """
        if n == "fptypes":
            return "typename"
        nns = self.get_ns(ns, n)
        if nns:
            nn = splitns(n)[1]
            if nn in self.namespace_dict[nns].enums:
                return re.sub(
                    r"(?<!daal::)algorithms::", r"daal::algorithms::", nns + "::" + nn
                )
            return "unknown_" + nns + "_class_" + n
        return "unknown_" + n

    ###############################################################################
    def get_class_for_typedef(self, ns, cls, td):
        """
        Find the Result type for given algorithm in the C++ class and namespace hierachy.
        Strips off potential SharedPtr def.
        Note: we assume there are no typedefs *Type outside classes
        """
        if ns not in self.namespace_dict or cls not in self.namespace_dict[ns].classes:
            return None
        if td not in self.namespace_dict[ns].classes[cls].typedefs:
            # Note: we assume there are no typedefs *Type outside classes
            for parent in self.namespace_dict[ns].classes[cls].parent:
                res = self.get_class_for_typedef(
                    self.get_ns(ns, parent), splitns(parent)[1], td
                )
                if res:
                    return res
            return None

        res = self.namespace_dict[ns].classes[cls].typedefs[td]
        tmp = splitns(res)
        ret = None
        if res.endswith("Type"):
            # Note: we assume there are no typedefs *Type outside classes
            assert res.endswith(td)
            ret = self.get_class_for_typedef(
                self.get_ns(ns, tmp[0]), splitns(tmp[0])[1], tmp[1]
            )
            if not ret and "<" in tmp[0]:
                n = tmp[0].split("<", 1)[0]
                ret = self.get_class_for_typedef(
                    self.get_ns(ns, n), splitns(n)[1], tmp[1]
                )
        else:
            ret = tmp
        if ret and ret[1] not in self.namespace_dict[ret[0]].classes and "<" in ret[1]:
            # Probably a template, sigh
            # For now, let's just cut off the template paramters.
            # Let's hope we don't need anything more sophisticated
            # (like if there are actually specializations...)
            c = ret[1].split("<", 1)[0]
            n = self.get_ns(ns, c)
            ret = (n, splitns(c)[1]) if n else None
            # ret = (self.get_ns(ns, res.split('<')[0]), tmp[1])
        return ret

    ###############################################################################
    def get_expand_attrs(self, ns, cls, attr):
        """
        Find enum type for attributes in "attr" dict of class "cls" and returns
        2-tuple of lists of tuples
          [0] list of members it can map to a hltype
          (namespace, member/value name, type of attr)
          [1] list of members it can not map tp hltype (namespace, attribute)
        Only standard data types and those with typemaps can be mapped to a hltype.
        """
        assert "members" not in attr, "get_expand_attrs is not supported for members"
        attrs = self.get_all_attrs(ns, cls, attr)
        explist = []
        ignlist = []
        for i in attrs:
            inp = splitns(i)[1]
            ins = self.get_ns(ns, i)
            assert ins
            assert ins in self.namespace_dict
            assert inp in self.namespace_dict[ins].enums
            if ignored(ns, "::".join([ins, inp])):
                continue
            hlt = self.to_hltype(ns, attrs[i])
            if hlt:
                if hlt[1] in ["stdtype", "enum", "class"]:
                    for e in self.namespace_dict[ins].enums[inp]:
                        if not any(e in x for x in explist) and not ignored(ins, e):
                            if type(attrs[i]) in [list, tuple]:
                                explist.append(
                                    (
                                        ins,
                                        e,
                                        hlt[0],
                                        attrs[i][1],
                                        self.namespace_dict[ins].enums[inp][e][1],
                                    )
                                )
                            else:
                                explist.append(
                                    (
                                        ins,
                                        e,
                                        hlt[0],
                                        None,
                                        self.namespace_dict[ins].enums[inp][e][1],
                                    )
                                )
                else:
                    print("// Warning: ignoring " + ns + " " + str(hlt))
                    ignlist.append((ins, i))
            else:
                print(
                    "// Warning: could not find hlt for"
                    " " + ns + " " + cls + " " + i + " " + str(attrs[i]) + ". Ignored."
                )
                ignlist.append((ins, i))
        return (explist, ignlist)

    ###############################################################################
    def prepare_resultmaps(self, ns):
        """
        Prepare info about typedefs for Result type of given namespace.
        Uses target language-specific defines/functions
          - native_type: returns native representation of its argument
          - TMGC(n): deals with GC(refcounting for given number of references (R)
        Looks up return type and then target-language
        independently creates lists of its content.
        We have not yet added support for 'get_methods'.
        """
        jparams = {}
        res = self.get_class_for_typedef(ns, "Batch", "ResultType")
        if not res and "Result" in self.namespace_dict[ns].classes:
            res = (ns, "Result")
        if res and "_".join(res) not in self.done:
            self.done.append("_".join(res))
            attrs = self.get_expand_attrs(res[0], res[1], "gets")
            if attrs and attrs[0]:
                jparams = {
                    "class_type": "daal::" + res[0] + "::" + res[1] + "Ptr",
                    "enum_gets": attrs[0],
                    "named_gets": [],
                    "get_methods": [],
                }
            else:
                print("// Warning: could not determine Result attributes for " + ns)
        elif res:
            jparams = {
                "class_type": "daal::" + res[0] + "::" + res[1] + "Ptr",
            }
        elif ns not in no_warn or "Result" not in no_warn[ns]:
            print("// Warning: no result found for " + ns)
        return jparams

    ###############################################################################
    def prepare_modelmaps(self, ns, mname="Model"):
        """
        Return string from typemap_wrapper_template for given Model.
        uses entries from 'gets' in Model class def to fill 'named_gets'.
        It also fills 'get_methods' for getters which require arguments.
        """
        jparams = {}
        if mname in self.namespace_dict[ns].classes:
            model = self.namespace_dict[ns].classes[mname]
            jparams = {
                "class_type": "daal::" + ns + "::ModelPtr",
                "enum_gets": [],
                "named_gets": [],
                "get_methods": [],
                "parent": model.parent,
            }
            huhu = self.get_all_attrs(ns, mname, "gets")
            for g in huhu:
                # We have a few get-methods accepting parameters, we map them separately
                if type(huhu[g]) in [list, tuple] and len(huhu[g]) > 2:
                    rtyp, ptyp, pnm = huhu[g]
                    gn = splitns(g)[1].replace("get", "")
                    if "::" in rtyp:
                        tns, ttyp = splitns(rtyp)
                        rtyp = "::".join(["daal::" + self.get_ns(ns, rtyp), ttyp])
                    jparams["get_methods"].append((rtyp, gn, ptyp, pnm))
                else:
                    if type(huhu[g]) in [list, tuple]:
                        rtyp, sfx = huhu[g]
                    else:
                        rtyp = huhu[g]
                        sfx = ""
                    if not any(
                        rtyp.endswith(x)
                        for x in [
                            "SerializationTag",
                        ]
                    ):
                        gn = splitns(g)[1].replace("get", "")
                        if not any(gn == x[1] for x in jparams["named_gets"]):
                            typ = re.sub(
                                r"(?<!daal::)data_management",
                                r"daal::data_management",
                                rtyp,
                            )
                            jparams["named_gets"].append((typ, gn, sfx))
        return jparams

    ###############################################################################
    def expand_typedefs(self, ns):
        """
        We expand all typedefs in classes/namespaces without recursing
        to outer scopes or namespaces.
        """

        def expand_td(typedefs):
            done = 1
            while done != 0:
                done = 0
                for td1 in typedefs:
                    for td2 in typedefs:
                        if td1 != td2 and (td1 + "::") in typedefs[td2]:
                            typedefs[td2] = typedefs[td2].replace(td1, typedefs[td1])
                            done += 1

        expand_td(self.namespace_dict[ns].typedefs)
        for c in self.namespace_dict[ns].classes:
            expand_td(self.namespace_dict[ns].classes[c].typedefs)

    ###############################################################################
    def order_iargs(self, tmp_input_args):
        """
        We have to put the intput args into the "right" order.
        , e.g. start with data then model, then whatever else
        """
        ordered = [
            "data",
            "model",
            "labels",
            "dependentVariable",
            "dependentVariables",
            "tableToFill",
            "dataForPruning",
            "dependentVariablesForPruning",
            "labelsForPruning",
            "inputArgument",
        ]
        input_args = []
        for arg in ordered:
            for i in tmp_input_args:
                if i.name.endswith(arg):
                    input_args.append(i)
        for i in enumerate(tmp_input_args):
            if not any(i[1].name.endswith(x) for x in ordered):
                input_args.append(i[1])
        return input_args

    ###############################################################################
    def get_template_specializations(self, ns, cls):
        res = []
        pat = cls + "<"
        for c in self.namespace_dict[ns].classes:
            if c.startswith(pat):
                res.append((ns, self.namespace_dict[ns].classes[c]))
        return res

    ###############################################################################
    def get_all_parameter_classes(self, ns):
        res = []
        for c in self.namespace_dict[ns].classes:
            if any(
                re.match(r"{}(<.+>)?$".format(x), c)
                for x in ["Batch", "Online", "Distributed"]
            ):
                p = self.get_class_for_typedef(ns, c, "ParameterType")
                if p and p not in res:
                    res.append((p[0], self.namespace_dict[p[0]].classes[p[1]]))
                    t = self.get_template_specializations(*p)
                    if t:
                        res += t
        return res

    ###############################################################################
    def prepare_hlwrapper(self, ns, mode, func, no_dist, no_stream):
        """
        Prepare data structures for generating high level wrappers.

        This is main function for generating high level wrappers.
        Here we prepare and return the data structure
        from which wrappers can be generated.
        The information is extracted from self.namespace_dict.

        We first extract template parameters and setup a generic structure/array.
        The generated array holds the general template spec
        in entry [0] and specializations in the following entires (if exist).

        We then extract input arguments.
        The resulting arrays begin with required inputs followed by optional inputs.
        We get the input arguments by expanding the enums
        from the set methods in the Input class (InputType).
        Each value in the enum becomes a separate input arguments.

        Next we extract parameters - required and optional separately.
        Each member of parameter_type becomes a separate arguments.
        Default values are handled in the lower C++ level.
        We provide "generic" default values like -1, NULL and "" for *all* parameters.
        These values cannot be reasonably used by the user.

        Next we extract type-map (native_type) information for Model and Result types.
        """
        if (
            mode in self.namespace_dict[ns].classes
            and self.namespace_dict[ns].classes[mode].template_args
        ):
            ins = (
                splitns(self.namespace_dict[ns].classes[mode].typedefs["InputType"])[0]
                if "InputType" in self.namespace_dict[ns].classes[mode].typedefs
                else ns
            )
            jparams = {
                "ns": ns,
                "algo": func,
                "template_decl": OrderedDict(),
                "template_spec": [],
                "template_args": [],
                "params_opt": [],
                "params_req": [],
                "params_get": "parameter",
                "params_templ": {},
                "opt_params": [],
                "s1": "step1Local",
                "s2": "step2Master",
            }
            # at this point required parameters need to be
            # explictly/manually provided in wrappers.required
            params_req = (
                [mk_var(x[0], x[1], algo=func) for x in required[ns]]
                if ns in required
                else []
            )
            if self.namespace_dict[ns].classes[mode].template_args:
                jparams["params_templ"] = {
                    "template_decl": OrderedDict(
                        [
                            (
                                t[0],
                                {
                                    "template_decl": self.get_tmplarg(ns, t[1]),
                                    "values": self.get_values(ns, t[1]),
                                    "default": t[2].replace(
                                        "DAAL_ALGORITHM_FP_TYPE", "double"
                                    ),
                                },
                            )
                            for t in self.namespace_dict[ns].classes[mode].template_args
                        ]
                    ),
                    "template_args": [
                        mk_var(
                            t[0],
                            "std::string&",
                            "const",
                            t[2].replace("DAAL_ALGORITHM_FP_TYPE", "double"),
                            algo=func,
                            doc=t[3],
                        )
                        for t in self.namespace_dict[ns].classes[mode].template_args
                    ],
                }

            # A parameter can be a specialized template. Sigh.
            # we need to provide specialized template classes for them.
            # We do this by creating a list for templates,
            # for specializations the list-length is >1 and the first entry
            #  is the base/"real" template spec, following entries are specializations.
            # At some point we might have other things that
            # influence this (like result or input).
            # for now, we only check for Parameter specializations

            param_classes = self.get_all_parameter_classes(ns)
            all_params = OrderedDict()
            opt_params = {}
            for p in param_classes:
                parms = self.get_all_attrs(p[0], p[1].name, "members", ns)
                assert "::" not in p[1].name
                # hack: we need to use fully qualified enum values,
                # proper solution would find enum...
                pcls = re.sub(r"(\w+\s*>)", r"daal::{}::\1", p[1].name).format(p[0])
                # We need to rename the template args,
                # since they might shadow the algorithm's template args
                # (like fptype, Method)
                if p[1].template_args and not p[1].partial:
                    # let's format the class name as the argument including template args
                    nm = "{}::{}<{}>".format(
                        p[0], pcls, ", ".join([x[0] + "_" for x in p[1].template_args])
                    )
                else:
                    nm = "{}::{}".format(p[0], pcls)
                    if p[1].template_args:
                        for x in p[1].template_args:
                            nm = nm.replace(x[0], x[0] + "_")
                if nm not in opt_params:
                    opt_params[nm] = (
                        (
                            [[x[0] + "_", x[1], x[2]] for x in p[1].template_args]
                            if p[1].template_args
                            else False
                        ),
                        [splitns(x)[1] for x in parms],
                    )
                for a in parms:
                    if a not in all_params:
                        all_params[a] = parms[a]

            bcls = "::".join([ns, "Batch"])
            if bcls in no_constructor:
                # Special mode were we have no parameters/constructor, but a create method
                all_params = no_constructor[bcls]
            elif len(all_params) >= 1:
                # If we have parameters,
                # check if we have an accessor func or a member 'parameter'
                p = self.get_all_attrs(ns, "Batch", "members")
                if not p or not any(x.endswith("parameter") for x in p):
                    jparams["params_get"] = "parameter()"
            else:
                # No parameters found
                if ns not in no_warn or "ParameterType" not in no_warn[ns]:
                    print("// Warning: no parameters found for " + ns)
                jparams["params_get"] = None

            for p in all_params:
                pns, tmp = splitns(p)
                if tmp is not None and not tmp.startswith("_") and not ignored(pns, tmp):
                    llt = self.to_lltype(p, all_params[p][0])
                    hlt = self.to_hltype(pns, llt)
                    if hlt and hlt[1] in ["stdtype", "enum", "class"]:
                        (hlt, hlt_type, hlt_ns) = hlt
                        if hlt_type == "enum":
                            thetype = hlt_ns + "::" + llt.rsplit("::", 1)[-1]
                            if ns in result_to_compute.keys():
                                if (
                                    result_to_compute[ns].rsplit("::", 1)[-1]
                                    in self.namespace_dict[get_parent(ns)].enums.keys()
                                ):
                                    thetype = result_to_compute[ns]
                        else:
                            thetype = hlt if hlt else all_params[p]
                        if thetype is not None:
                            thetype = re.sub(
                                r"(?<!daal::)algorithms::", r"daal::algorithms::", thetype
                            )
                            doc = all_params[p][1]
                            if any(tmp == x.name for x in params_req):
                                v = mk_var(tmp, thetype, "const", algo=func, doc=doc)
                                jparams["params_req"].append(v)
                            else:
                                prm = tmp
                                dflt = (
                                    defaults[pns][prm]
                                    if pns in defaults and prm in defaults[pns]
                                    else True
                                )
                                v = mk_var(
                                    prm, thetype, "const", dflt, algo=func, doc=doc
                                )
                                jparams["params_opt"].append(v)
                        else:
                            print(
                                "// Warning: do not know what to do with " + pns + " :"
                                " " + p + "(" + all_params[p] + ")"
                            )
                    else:
                        print(
                            "// Warning: parameter member " + p + " of " + pns + " "
                            "is no stdtype, no enum and not a DAAl class. Ignored."
                        )

            # we now prepare the optional arguments per Parameter class,
            # so that we can generate
            # specialized init_parameter()'s
            for pcls in opt_params:
                val = opt_params[pcls]
                tmp = []
                for opt in jparams["params_opt"]:
                    if opt.name in val[1]:
                        tmp.append(opt)
                jparams["opt_params"].append((pcls, val[0], tmp))

            # Now let's get the input arguments (provided to input class/object of algos)
            tmp_input_args = []
            inp = self.get_class_for_typedef(ns, "Batch", "InputType")
            if not inp and "Input" in self.namespace_dict[ns].classes:
                inp = (ns, "Input")
            if inp:
                expinputs = self.get_expand_attrs(inp[0], inp[1], "sets")
                reqi = 0
                for ins, iname, itype, optarg, doc in expinputs[0]:
                    tmpi = iname
                    if tmpi and not ignored(ns, tmpi):
                        if ns in defaults and tmpi in defaults[ns]:
                            i = len(tmp_input_args)
                            dflt = defaults[ns][tmpi]
                        else:
                            i = reqi
                            reqi += 1
                            dflt = None
                        if "::NumericTablePtr" in itype:
                            # ns in has_dist and \
                            # iname in has_dist[ns]['step_specs'][0].inputnames or \
                            # iname in ['data', 'labels',\
                            #           'dependentVariable', 'tableToFill']:
                            itype = "data_or_file &"
                        ins = re.sub(
                            r"(?<!daal::)algorithms::", r"daal::algorithms::", ins
                        )
                        tmp_input_args.insert(
                            i,
                            mk_var(
                                ins + "::" + iname,
                                itype,
                                "const",
                                dflt,
                                inpt=True,
                                algo=func,
                                doc=doc,
                            ),
                        )
            else:
                print("// Warning: no input type found for " + ns)

            # We have to bring the input args into the "right" order
            jparams["input_args"] = self.order_iargs(tmp_input_args)
            # we will need something more sophisticated
            # if the interesting parent class
            # is not a direct parent (a grand-parent for example)
            ifcs = []
            for i in self.namespace_dict[ns].classes[mode].parent:
                pns = self.get_ns(ns, i, attrs=["classes"])
                if pns:
                    p = "::".join([pns.replace("algorithms::", ""), splitns(i)[1]])
                    if p in ifaces:
                        ifcs.append(cpp2hl(p))
            jparams["iface"] = ifcs if ifcs else [None]
        else:
            jparams = {}

        # here we know parameters, inputs etc for each
        # let's store this
        fcls = "::".join([ns, mode])
        retjp = {
            "params": jparams,
            "model_typemap": self.prepare_modelmaps(ns),
            "result_typemap": self.prepare_resultmaps(ns),
            "create": no_constructor[fcls] if fcls in no_constructor else "",
            "add_setup": add_setup[ns] if ns in add_setup else None,
            "add_get_result": True if ns in add_get_result else None,
        }
        if not no_dist and ns in has_dist:
            retjp["dist"] = has_dist[ns]
            retjp["distributed"] = mk_var(
                "distributed",
                "bool",
                dflt=True,
                algo=func,
                doc="enable distributed computation (SPMD)",
            )
        else:
            retjp["distributed"] = mk_var()
        if (
            not no_stream
            and "Online" in self.namespace_dict[ns].classes
            and not ns.endswith("pca")
        ):
            retjp["streaming"] = mk_var(
                "streaming", "bool", dflt=True, algo=func, doc="enable streaming"
            )
        else:
            retjp["streaming"] = mk_var()
        return {ns + "::" + mode: retjp}

    def prepare_model_hierachy(self, cfg):
        """
        Create a dict which lists all child classes for each Model.
        Flatens the full hierachy for each Model.
        """
        model_hierarchy = defaultdict(lambda: [])
        for ns in cfg:
            c = cfg[ns]
            if c["model_typemap"]:
                for parent in c["model_typemap"]["parent"]:
                    if "algorithms::" not in parent:
                        parent = "algorithms::" + parent
                    if "daal::" not in parent:
                        parent = "daal::" + parent
                    if not parent.endswith("Ptr"):
                        parent = parent + "Ptr"
                    model_hierarchy[parent].append(c["model_typemap"]["class_type"])

        done = 0 if len(model_hierarchy) > 0 else 1
        # We now have to expand so that each ancestor holds a list of all its decendents
        while done == 0:
            done = 0
            adds = {}
            for parent in model_hierarchy:
                adds[parent] = []
                c = model_hierarchy[parent]
                for child in c:
                    if child in model_hierarchy:
                        gc = model_hierarchy[child]
                        for gchild in gc:
                            if gchild not in c:
                                adds[parent].append(gchild)
                                done = 1
            for m in adds:
                model_hierarchy[m] = adds[m] + model_hierarchy[m]

        for m in model_hierarchy:
            ns = splitns(m)[0].replace("daal::", "") + "::Batch"
            if ns in cfg:
                cfg[ns]["model_typemap"]["derived"] = model_hierarchy[m]

    def hlapi(self, version, no_dist=False, no_stream=False):
        """
        Generate high level wrappers for namespaces allowed by wrap_algo(ns, version).

        First extract the namespaces we really want, e.g. ignore NN.

        Then we expand typedefs on class and namespace levels.

        Then generate maps for each algo mapping string arguments to C++ enum values.

        Next prepares parsed data for code generation (e.g. setting up dicts for jinja).

        Finally generates
          - Result/Model classes and their properties
          - algo-wrappers
          - initialization

        We generate strings for a C++ header, a C++ file and a cython file.
        """
        dtypes = ""
        algoconfig = {}

        algos = [x for x in self.namespace_dict if wrap_algo(x, version)]

        # First expand typedefs
        for ns in algos:
            self.expand_typedefs(ns)
        # Next, extract and prepare the data (input, parameters, results, template spec)
        for ns in algos:
            if not ignored(ns):
                nn = ns.split("::")
                if nn[0] == "daal":
                    if nn[1] == "algorithms":
                        func = "_".join(nn[2:])
                    else:
                        func = "_".join(nn[1:])
                elif nn[0] == "algorithms":
                    func = "_".join(nn[1:])
                else:
                    func = "_".join(nn)
                algoconfig.update(
                    self.prepare_hlwrapper(ns, "Batch", func, no_dist, no_stream)
                )

        self.prepare_model_hierachy(algoconfig)

        # and now we can finally generate the code
        wg = wrapper_gen(algoconfig, {cpp2hl(i): ifaces[i] for i in ifaces})
        cpp_map, cpp_begin, cpp_end, pyx_map, pyx_begin, pyx_end = (
            "",
            "",
            '#define NO_IMPORT_ARRAY\n#include "daal4py_cpp.h"\n',
            "",
            "",
            "",
        )

        for ns in algos:
            if (
                ns.startswith("algorithms::")
                and not ns.startswith("algorithms::neural_networks")
                and self.namespace_dict[ns].enums
            ):
                cpp_begin += "static str2i_map_t s2e_" + ns.replace("::", "_") + " =\n{\n"
                for e in self.namespace_dict[ns].enums:
                    for v in self.namespace_dict[ns].enums[e]:
                        vv = ns + "::" + v
                        cpp_begin += " " * 4 + '{"' + v + '", daal::' + vv + "},\n"
                cpp_begin += "};\n\n"
                # For enums that are used to access KeyValueDataCollections
                # we need an inverse map value->string. Note this is enum-specific,
                # we cannot have one for the ns because
                # they might have duplicate values
                for e in self.namespace_dict[ns].enums:
                    enm = "::".join([ns, e])
                    if enm in enum_maps:
                        cpp_begin += (
                            "static i2str_map_t e2s_"
                            + ns.replace("::", "_")
                            + "_"
                            + enum_maps[enm]
                            + " =\n{\n"
                        )
                        for v in self.namespace_dict[ns].enums[e]:
                            vv = ns + "::" + v
                            cpp_begin += " " * 4 + "{daal::" + vv + ', "' + v + '"},\n'
                        cpp_begin += "};\n\n"

        for a in algoconfig:
            (ns, algo) = splitns(a)
            if algo.startswith("Batch"):
                tmp = wg.gen_wrapper(ns, algo)
                if tmp:
                    cpp_map += tmp[0]
                    cpp_begin += tmp[1]
                    cpp_end += tmp[2]
                    pyx_map += tmp[3]
                    pyx_begin += tmp[4]
                    pyx_end += tmp[5]
                    dtypes += tmp[6]

        hds = wg.gen_headers()
        fts = wg.gen_footers(
            no_dist,
            no_stream,
            algos,
            version,
            [x for x in has_dist if has_dist[x]["pattern"] == "dist_custom"],
        )
        pyx_end += fts[1]

        # Finally combine the different sections and return the 3 strings
        return (
            hds[0] + cpp_map + cpp_begin + fts[2] + "\n#endif",
            cpp_end,
            hds[1] + pyx_map + pyx_begin + pyx_end,
        )


###############################################################################
###############################################################################
###############################################################################
###############################################################################


def gen_daal4py(dalroot, outdir, version, warn_all=False, no_dist=False, no_stream=False):
    global no_warn
    if warn_all:
        no_warn = {}
    orig_path_candidates = [
        jp(dalroot, "include", "dal"),
        jp(dalroot, "Library", "include", "dal"),
        jp(dalroot, "include"),
    ]
    for candidate in orig_path_candidates:
        if isdir(candidate):
            orig_path = candidate
            break
    assert os.path.isfile(jp(orig_path, "algorithms", "algorithm.h")) and os.path.isfile(
        jp(orig_path, "algorithms", "model.h")
    ), (
        "Path/$DALROOT '"
        + orig_path
        + "' doesn't seem host oneDAL headers. Please provide correct dalroot."
    )
    head_path = jp("build", "include")
    algo_path = jp(head_path, "algorithms")
    rmtree(head_path, ignore_errors=True)
    copytree(orig_path, head_path)
    formatfile = jp("src", ".clang-format")
    for dirpath, dirnames, filenames in os.walk(algo_path):
        for filename in filenames:
            call(
                [
                    shutil.which("clang-format"),
                    "-i",
                    jp(dirpath, filename),
                    "-style=file:" + formatfile,
                ]
            )
    iface = cython_interface(algo_path)
    iface.read()
    print("Generating sources...")
    cpp_h, cpp_cpp, pyx_file = iface.hlapi(iface.version, no_dist, no_stream)

    # 'ridge_regression', parametertype is a template without any need
    with open(jp(outdir, "daal4py_cpp.h"), "w") as f:
        f.write(cpp_h)
    with open(jp(outdir, "daal4py_cpp.cpp"), "w") as f:
        f.write(cpp_cpp)
    with open(jp("src", "gettree.pyx"), "r") as f:
        pyx_gettree = f.read()

    pyx_gbt_model_builder = ""
    pyx_gbt_generators = ""
    pyx_log_reg_model_builder = ""
    pyx_gettree = ""
    if (
        "algorithms::gbt::classification" in iface.namespace_dict
        and "ModelBuilder"
        in iface.namespace_dict["algorithms::gbt::classification"].classes
    ):
        with open(jp("src", "gbt_model_builder.pyx"), "r") as f:
            pyx_gbt_model_builder = f.read()
    if (
        "algorithms::logistic_regression" in iface.namespace_dict
        and "ModelBuilder"
        in iface.namespace_dict["algorithms::logistic_regression"].classes
    ):
        with open(jp("src", "log_reg_model_builder.pyx"), "r") as f:
            pyx_log_reg_model_builder = f.read()
    if "algorithms::tree_utils" in iface.namespace_dict:
        with open(jp("src", "gettree.pyx"), "r") as f:
            pyx_gettree = f.read()

    with open(jp(outdir, "daal4py_cy.pyx"), "w") as f:
        f.write(pyx_file)
        f.write(pyx_gettree)
        f.write(pyx_gbt_model_builder)
        f.write(pyx_log_reg_model_builder)
        f.write(pyx_gbt_generators)
