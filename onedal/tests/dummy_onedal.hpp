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

///////////////////////////// Fake oneDAL Algorithm ///////////////////////
// These aspects fake the necessary characteristics of a oneDAL algorithm
// They forego the indirections used with impl_ attributes characteristic
// of the oneDAL codebase, and only show the necessary APIs. It is also as
// minimal as possible, dropping some required setters/getters for brevity.

// These aspects are created in the algorithm's common.hpp
namespace task {
    struct generate {};
    using by_default = generate;
}

namespace method {
    struct dense {};
    using by_default = dense;
}

template <typename Float = float,
          typename Method = method::by_default,
          typename Task = task::by_default>
class descriptor : public detail::descriptor_base<Task>

public:
    using float_t = Float;
    using method_t = Method;
    using task_t = Task;

    descriptor() = default;

}

// These aspects are created in the algorithm's train_types.hpp

template <typename Task = task::by_default>
class train_input : public base {

public:
    using task_t = Task

    train_input(const table& data)

}

template <typename Task = task::by_default>
class train_result {

public:
    using task_t = Task;
        
    train_result();

    const &

}

// These aspects are created in the algorithm's detail/train_ops.hpp

template <typename Descriptor = >
struct train_ops {
    using float_t = typename Descriptor::float_t;
    using method_t = typename Descriptor::method_t;
    using task_t = typename Descriptor::task_t;

    template <typename Context>
    train_result<task_t> operator()(const Context& ctx,
                                    const Descriptor& desc,
                                    const input_t& input) const {
}
}
// These aspect are create in the algorithm's infer_types.hpp
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

template <typename Task = task::by_default>
class infer_result {

public:
    using task_t = Task;

    infer_result(){}

    const table& get_data(){return this->data;}

    // attribute usually hidden in an infer_result_impl class
    table data;

}

// These aspects are created in the algorithm's detail/infer_ops.hpp
template <typename Descriptor>
struct infer_ops {
    using float_t = typename Descriptor::float_t;
    using method_t = typename Descriptor::method_t;
    using task_t = typename Descriptor::task_t;
    using input_t = infer_input<task_t>;
    using result_t = infer_result<task_t>;

template <typename Context>
    auto operator()(const Context& ctx, const Descriptor& desc, const input_t& input) const {
    }
}
///////////////////////////// Fake oneDAL Algorithm ///////////////////////

}
