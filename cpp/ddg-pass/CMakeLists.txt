# DDGPass — LLVM plug-in: per-basic-block data-dependence graph extractor
#
# Minimum LLVM version: 15  (new-pass-manager plugin API required)
# Primary targets:      LLVM 18, LLVM 19
# Supported platforms:  Linux (x86-64, AArch64), macOS (x86-64, Apple Silicon)
# C++ standard:         C++17
#
# Quick build:
#   mkdir build && cd build
#   cmake -DLLVM_DIR=$(llvm-config --cmakedir) ..
#   cmake --build . -j$(nproc)
#   # Produces: build/DDGPass.so  (Linux) / build/DDGPass.dylib  (macOS)

cmake_minimum_required(VERSION 3.20)
project(DDGPass C CXX)

# ---------------------------------------------------------------------------
# LLVM
# ---------------------------------------------------------------------------
# LLVM's CMake config uses SameMajorVersion compatibility, which causes
# find_package(LLVM <N>) to reject any LLVM whose major version != N.
# Find any installed LLVM and enforce the minimum version bound manually.
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM_DIR: ${LLVM_DIR}")
message(STATUS "LLVM include dirs: ${LLVM_INCLUDE_DIRS}")

if(LLVM_PACKAGE_VERSION VERSION_LESS "15")
  message(FATAL_ERROR
    "LLVM 15 or later required; found LLVM ${LLVM_PACKAGE_VERSION}. "
    "Install LLVM 15+ and re-run.")
endif()

# Match LLVM's C++ standard and RTTI settings exactly.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# add_llvm_pass_plugin() does not add LLVM's headers or definitions itself.
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

include(HandleLLVMOptions)
if(NOT LLVM_ENABLE_RTTI)
  add_compile_options(-fno-rtti)
endif()

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM OPTIONAL)

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
if(COMMAND add_llvm_pass_plugin)
  # LLVM CMake helper available (LLVM 15+).
  add_llvm_pass_plugin(DDGPass DDGPass.cpp)
else()
  # Fallback: plain MODULE library.
  add_library(DDGPass MODULE DDGPass.cpp)
  target_include_directories(DDGPass PRIVATE ${LLVM_INCLUDE_DIRS})
  target_compile_definitions(DDGPass PRIVATE ${LLVM_DEFINITIONS})
  if(TARGET LLVM)
    target_link_libraries(DDGPass PRIVATE LLVM)
  else()
    llvm_map_components_to_libnames(LLVM_LIBS core support passes analysis)
    target_link_libraries(DDGPass PRIVATE ${LLVM_LIBS})
  endif()
endif()

target_compile_features(DDGPass PRIVATE cxx_std_17)

# Ensure the plugin entry symbol is visible to dlopen/LoadLibrary and that
# the library does not carry the "lib" prefix (DDGPass.so, not libDDGPass.so).
set_target_properties(DDGPass PROPERTIES
  CXX_VISIBILITY_PRESET     default
  VISIBILITY_INLINES_HIDDEN OFF
  POSITION_INDEPENDENT_CODE ON
  PREFIX                    ""
)

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS DDGPass
  LIBRARY DESTINATION lib/ddg-pass
)
