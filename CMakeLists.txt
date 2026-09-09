cmake_minimum_required(VERSION 3.18)
project(sew_mimic_exact_core LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_exact_sew_core MODULE cpp/exact_sew_core.cpp)
target_compile_features(_exact_sew_core PRIVATE cxx_std_17)
target_compile_definitions(_exact_sew_core PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES)

if(MSVC)
  target_compile_options(_exact_sew_core PRIVATE /O2 /permissive-)
  target_link_options(_exact_sew_core PRIVATE /LTCG)
else()
  target_compile_options(_exact_sew_core PRIVATE -O3)
endif()

install(TARGETS _exact_sew_core DESTINATION sew_mimic/exact)
