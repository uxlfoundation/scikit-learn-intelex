/*******************************************************************************
* Copyright Contributors to the oneDAL Project
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/


#include "onedal/common.hpp"
#include "onedal/version.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/homogen.hpp"

namespace py = pybind11;

namespace oneapi::dal {

namespace dummy {

////////////////////////// Dummy oneDAL Algorithm /////////////////////////
// These aspects fake the necessary API characteristics of a oneDAL 
// algorithm. This example foregoes the indirections used with impl_ 
// attributes characteristic of the oneDAL codebase and only show the
// necessary APIs. It is also as minimal as possible, dropping some 
// required setters/getters for brevity. It also violates some rules with
// respect to protected/private, attributes, and compile time type checking.
//
// Files which are normally separated in oneDAL for clarity are merged here
// to provide an overview of what is necessary for interaction in sklearnex.
//
// To support oneDAL offloading, task, method and descriptor structs need
// to be defined from the algorithm's common.hpp.
//
// For various modes (e.g. training, inference), the requisite functors and
// result data structs need to be defined. Usually this is in *_types.hpp.
// For example, a 'compute' algorithm would have a compute_types.hpp
//
// Usually these aspects are all made available via the algorithm's header
// file located in oneapi/dal/algo.
//
// This should act as a guide for where to look and what to reference in
// oneDAL for making a pybind11 interface.

/////////////////////////////// common.hpp ////////////////////////////////

namespace task {
    // tasks can be arbitrarily named, ``by_default`` must be defined.
    struct generate {};
    using by_default = generate;
}

namespace method {
    // methods can be arbirarily named, though this will be used in the
    // python onedal estimator as a parameter
    struct dense {};
    using by_default = dense;
}

namespace detail {
// This is highly important for central use of train, compute, infer etc.
// but is not used in sklearnex (and must be included here).
struct descriptor_tag {};

}

template <typename Float = float,
          typename Method = method::by_default,
          typename Task = task::by_default>
class descriptor : public base {

public:
    using tag_t = descriptor_tag;
    using float_t = Float;
    using method_t = Method;
    using task_t = Task;

    descriptor(): constant(0.0) {}

    double get_constant() const {
        return this->constant;
    }

    auto& set_constant(double value) {
        this->constant = value;
        return *this;
    }

    // normally this attribute is hidden in another struct
    double constant

}

/////////////////////////////// common.hpp ////////////////////////////////


///////////////////////////// train_types.hpp /////////////////////////////
template <typename Task = task::by_default>
class train_result {

public:
    using task_t = Task;
        
    train_result();

    const &

}

template <typename Task = task::by_default>
class train_input : public base {

public:
    using task_t = Task

    train_input(const table& data)

}

///////////////////////////// train_types.hpp /////////////////////////////


///////////////////////////// infer_types.hpp /////////////////////////////
template <typename Task = task::by_default>
class infer_result {

public:
    using task_t = Task;

    infer_result(){}

    const table& get_data(){return this->data;}

    // attribute usually hidden in an infer_result_impl class
    table data;

}

template <typename Task = task::by_default>
class infer_input : public base {

public:
    using task_t = Task;

    infer_input(const table& data, const table& constant): data(data), constant(constant) {}
    // setters and getters for ``data`` and ``model`` removed for brevity
    
    // attributes usually hidden in an infer_input_impl class
    table data;
    table constant;
}
///////////////////////////// infer_types.hpp /////////////////////////////


/////// THESE ARE PRIVATE STEPS REQUIRED FOR IT TO WORK WITH ONEDAL ///////
template <typename Context, typename Float, typename Method, typename Task, typename... Options>
struct infer_ops_dispatcher {
    infer_result<Task> operator()(const Context&,
                                    const descriptor_base<Task>& desc,
                                    const compute_input<Task>&) const;
};


template <typename Descriptor>
struct infer_ops {
    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = method::by_default;
    using input_t = infer_input<task_t>;
    using result_t = infer_result<task_t>;

    template <typename Context>
    auto operator()(const Context& ctx, const Descriptor& desc, const input_t& input) const {
        const auto result =
            infer_ops_dispatcher<Context, float_t, method_t, task_t>()(ctx, desc, input);
        return result;
    }


template <typename Descriptor>
struct infer_ops<Descriptor, dal::decision_forest::detail::descriptor_tag>
        : dal::decision_forest::detail::infer_ops<Descriptor> {};


////////////////////////// Dummy oneDAL Algorithm /////////////////////////

}
