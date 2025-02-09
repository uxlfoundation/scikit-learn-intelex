#===============================================================================
# Copyright 2021 Intel Corporation
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
#===============================================================================

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project("oneDAL python interface"
    LANGUAGES CXX
    HOMEPAGE_URL https://github.com/uxlfoundation/oneDAL
)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ADD_ONEDAL_RPATH "Adds oneDAL's file paths to the RPATH here" OFF)
message(STATUS "ADD_ONEDAL_RPATH:" ${ADD_ONEDAL_RPATH})

option(SKLEARNEX_GCOV "Compile with gcov" OFF)
message(STATUS "SKLEARNEX_GCOV:" ${SKLEARNEX_GCOV})

if(WIN32)
    # hint CMake to get python from PYTHON env. variable if defined
    if(DEFINED ENV{PYTHON})
        set(PYTHON_EXECUTABLE $ENV{PYTHON})
    endif()
    set(SDL_FLAGS "-GS -DynamicBase")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MD")
elseif(UNIX)
    set(SDL_FLAGS "-fstack-protector-strong -fPIC -D_FORTIFY_SOURCE=2 -Wformat -Wformat-security -fno-strict-overflow -fno-delete-null-pointer-checks")
    if(APPLE)
        set(ONEDAL_PY_LINK_OPTIONS "-fwrapv")
        set(CMAKE_INSTALL_RPATH "@loader_path/../../../")
    else()
        set(ONEDAL_PY_LINK_OPTIONS "-Wl,-z,noexecstack,-z,relro,-z,now,-fstack-protector-strong,-fno-strict-overflow,-fno-delete-null-pointer-checks,-fwrapv")
        if(ADD_ONEDAL_RPATH)
            set(CMAKE_INSTALL_RPATH "${oneDAL_LIBRARY_DIR}:$ORIGIN/../../../")
        else()
            set(CMAKE_INSTALL_RPATH "$ORIGIN/../../../")
        endif()
    endif()
    set(WARNING_FLAGS "-Winit-self")
else()
    message(FATAL_ERROR "Unsupported system.")
endif()

set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${WARNING_FLAGS} ${SDL_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS} ${SDL_FLAGS}")

set(PYTHON_ONEDAL "${CMAKE_CURRENT_SOURCE_DIR}/../onedal/")

file(GLOB_RECURSE sources
    ${PYTHON_ONEDAL}/**/*.cpp
    ${PYTHON_ONEDAL}/*.cpp
)

message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}")

# find_package(oneDAL REQUIRED)
message(STATUS "oneDAL_INCLUDE_DIRS:" ${oneDAL_INCLUDE_DIRS})
message(STATUS "oneDAL_LIBRARY_DIR:" ${oneDAL_LIBRARY_DIR})
message(STATUS "PYTHON_INCLUDE_DIR:" ${PYTHON_INCLUDE_DIR})
message(STATUS "PYTHON_LIBRARY_DIR:" ${PYTHON_LIBRARY_DIR})
message(STATUS "NUMPY_INCLUDE_DIRS:" ${NUMPY_INCLUDE_DIRS})

if(IFACE STREQUAL "dpc")
    set(IFACE_IS_DPC ON)
else()
    set(IFACE_IS_DPC OFF)
endif()

if(IFACE STREQUAL "spmd_dpc")
    set(IFACE_IS_SPMD_DPC ON)
else()
    set(IFACE_IS_SPMD_DPC OFF)
endif()

if(IFACE_IS_SPMD_DPC)
    message(STATUS "MPI_INCLUDE_DIRS:" ${MPI_INCLUDE_DIRS})
    message(STATUS "MPI_LIBRARY_DIR:" ${MPI_LIBRARY_DIR})
endif()

message(STATUS "oneDAL_USE_PARAMETERS_LIB:" ${oneDAL_USE_PARAMETERS_LIB})

find_package(pybind11 REQUIRED)

if(IFACE STREQUAL "host")
    set(TARGET "_onedal_py_host")

    set(ONEDAL_LIBRARIES "")
    if(WIN32)
        list(APPEND ONEDAL_LIBRARIES "onedal_dll.${ONEDAL_MAJOR_BINARY}")
        list(APPEND ONEDAL_LIBRARIES "onedal_core_dll.${ONEDAL_MAJOR_BINARY}")
    elseif(APPLE)
        list(APPEND ONEDAL_LIBRARIES "onedal.${ONEDAL_MAJOR_BINARY}")
        list(APPEND ONEDAL_LIBRARIES "onedal_core.${ONEDAL_MAJOR_BINARY}")
        list(APPEND ONEDAL_LIBRARIES "onedal_thread.${ONEDAL_MAJOR_BINARY}")
    else()
        list(APPEND ONEDAL_LIBRARIES "-l:libonedal.so.${ONEDAL_MAJOR_BINARY}")
        list(APPEND ONEDAL_LIBRARIES "-l:libonedal_core.so.${ONEDAL_MAJOR_BINARY}")
        list(APPEND ONEDAL_LIBRARIES "-l:libonedal_thread.so.${ONEDAL_MAJOR_BINARY}")
    endif()

    if(oneDAL_USE_PARAMETERS_LIB)
        if(WIN32)
            list(APPEND ONEDAL_LIBRARIES "onedal_core_parameters_dll.${ONEDAL_MAJOR_BINARY}")
        elseif(APPLE)
            list(APPEND ONEDAL_LIBRARIES "onedal_parameters.${ONEDAL_MAJOR_BINARY}")
        else()
            list(APPEND ONEDAL_LIBRARIES "-l:libonedal_parameters.so.${ONEDAL_MAJOR_BINARY}")
        endif()
    endif()

    if(SKLEARNEX_GCOV)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
            if(WIN32)
                set(CMAKE_CXX_FLAGS "/clang:--coverage ${CMAKE_CXX_FLAGS}")
                list(APPEND ONEDAL_LIBRARIES "clang_rt.profile-x86_64.lib")
            else()
                set(CMAKE_CXX_FLAGS "--coverage ${CMAKE_CXX_FLAGS}")
                set(CMAKE_SHARED_LINKER_FLAGS "--coverage ${CMAKE_SHARED_LINKER_FLAGS}")
            endif()
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            set(CMAKE_CXX_FLAGS "--coverage ${CMAKE_CXX_FLAGS}")
            set(CMAKE_SHARED_LINKER_FLAGS "--coverage ${CMAKE_SHARED_LINKER_FLAGS}")
        else()
            message(WARNING "Code coverage will not be generated for target: host")
        endif()
    endif()

    list(APPEND COMPILE_DEFINITIONS "NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION")

elseif(IFACE_IS_DPC OR IFACE_IS_SPMD_DPC)

    if(IFACE_IS_SPMD_DPC)
        set(TARGET "_onedal_py_spmd_dpc")
    else()
        set(TARGET "_onedal_py_dpc")
    endif()

    # FIXME: icx>=2024.1 and pybind11 don't work correctly with IPO
    if(WIN32)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        set(CMAKE_CXX_FLAGS "-fsycl ${CMAKE_CXX_FLAGS}")
        
    endif()

    if(CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM")
        set(CMAKE_C_FLAGS "-fsycl ${CMAKE_C_FLAGS}")
    endif()

    set(ONEDAL_LIBRARIES "")
    if(WIN32)
        list(APPEND ONEDAL_LIBRARIES "onedal_dpc_dll.${ONEDAL_MAJOR_BINARY}")
        list(APPEND ONEDAL_LIBRARIES "onedal_core_dll.${ONEDAL_MAJOR_BINARY}")
    elseif(APPLE)
        list(APPEND ONEDAL_LIBRARIES "onedal_dpc.${ONEDAL_MAJOR_BINARY}")
        list(APPEND ONEDAL_LIBRARIES "onedal_core.${ONEDAL_MAJOR_BINARY}")
        list(APPEND ONEDAL_LIBRARIES "onedal_thread.${ONEDAL_MAJOR_BINARY}")
    else()
        list(APPEND ONEDAL_LIBRARIES "-l:libonedal_dpc.so.${ONEDAL_MAJOR_BINARY}")
        list(APPEND ONEDAL_LIBRARIES "-l:libonedal_core.so.${ONEDAL_MAJOR_BINARY}")
        list(APPEND ONEDAL_LIBRARIES "-l:libonedal_thread.so.${ONEDAL_MAJOR_BINARY}")
    endif()

    if(oneDAL_USE_PARAMETERS_LIB)
        if(WIN32)
            list(APPEND ONEDAL_LIBRARIES "onedal_core_parameters_dpc_dll.${ONEDAL_MAJOR_BINARY}")
        elseif(APPLE)
            list(APPEND ONEDAL_LIBRARIES "onedal_parameters_dpc.${ONEDAL_MAJOR_BINARY}")
        else()
            list(APPEND ONEDAL_LIBRARIES "-l:libonedal_parameters_dpc.so.${ONEDAL_MAJOR_BINARY}")
        endif()
    endif()

    if(SKLEARNEX_GCOV)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" AND WIN32)
            set(CMAKE_CXX_FLAGS "/clang:-Xarch_host /clang:--coverage ${CMAKE_CXX_FLAGS}")
            list(APPEND ONEDAL_LIBRARIES "clang_rt.profile-x86_64.lib")
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" AND UNIX)
            set(CMAKE_CXX_FLAGS "-Xarch_host --coverage ${CMAKE_CXX_FLAGS}")
            set(CMAKE_SHARED_LINKER_FLAGS "-Xarch_host --coverage ${CMAKE_SHARED_LINKER_FLAGS}")
        else()
            message(WARNING "Code coverage will not be generated for target: " ${IFACE})
        endif()
    endif()

    if(IFACE_IS_SPMD_DPC)
        set(MPI_LIBRARY ${MPI_LIBS})
    endif()

    list(APPEND COMPILE_DEFINITIONS
        "ONEDAL_DATA_PARALLEL"
        "NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION")

    if(IFACE_IS_SPMD_DPC)
        list(APPEND COMPILE_DEFINITIONS
             "ONEDAL_DATA_PARALLEL_SPMD")
    endif()

else()
    message(FATAL_ERROR "Invalid IFACE value: " ${IFACE})
endif()

pybind11_add_module(${TARGET} ${sources})

set(EXTERNAL_INCLUDE_DIRECTORIES
    ${PYTHON_ONEDAL}../
    ${oneDAL_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIR}
    ${NUMPY_INCLUDE_DIRS}
)

# TODO: remove this workaround when oneDAL CMake config is fixed
if(WIN32)
list(APPEND EXTERNAL_INCLUDE_DIRECTORIES $ENV{DALROOT}/Library/include/dal $ENV{DALROOT}/include/dal)
else()
list(APPEND EXTERNAL_INCLUDE_DIRECTORIES ${oneDAL_INCLUDE_DIRS}/dal)
endif()

set(EXTERNAL_LINK_DIRECTORIES ${PYTHON_LIBRARY_DIR} ${oneDAL_LIBRARY_DIR})
set(EXTERNAL_LINK_LIBRARIES ${ONEDAL_LIBRARIES})

if(IFACE_IS_SPMD_DPC)
list(APPEND EXTERNAL_LINK_LIBRARIES ${MPI_LIBRARY})
list(APPEND EXTERNAL_LINK_DIRECTORIES ${MPI_LIBRARY_DIR})
list(APPEND EXTERNAL_INCLUDE_DIRECTORIES ${MPI_INCLUDE_DIRS})
endif()

target_link_options(${TARGET} PRIVATE ${ONEDAL_PY_LINK_OPTIONS})
target_link_libraries(${TARGET} PUBLIC ${EXTERNAL_LINK_LIBRARIES})
target_compile_definitions(${TARGET} PUBLIC ${COMPILE_DEFINITIONS})
target_link_directories(${TARGET} PRIVATE ${EXTERNAL_LINK_DIRECTORIES})
target_include_directories(${TARGET} PRIVATE ${EXTERNAL_INCLUDE_DIRECTORIES})

message(STATUS "NUMPY_INCLUDE_DIRS:" ${NUMPY_INCLUDE_DIRS})

install(
  TARGETS ${TARGET}
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}
)
