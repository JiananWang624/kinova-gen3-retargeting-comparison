cmake_minimum_required(VERSION 3.18)
project(tiago_pro_7d_core LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_pro_7d_core MODULE cpp/pro_7d_core.cpp)
target_compile_features(_pro_7d_core PRIVATE cxx_std_17)

if(MSVC)
  target_compile_options(_pro_7d_core PRIVATE /O2 /permissive-)
else()
  target_compile_options(_pro_7d_core PRIVATE -O3)
endif()

install(TARGETS _pro_7d_core DESTINATION sew_mimic/pro)
