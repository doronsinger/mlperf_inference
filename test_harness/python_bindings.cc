#ifndef PYTHON_BINDINGS_H
#define PYTHON_BINDINGS_H

#include "system_under_test_c_api.h"
#include "test_harness.h"

#include "third_party/pybind/include/pybind11/pybind11.h"

PYBIND11_MODULE(mlpi_test_harness, m) {
  m.doc() = "MLPerf Inference test harness and traffic generator.";
  m.def("ConstructSUT", &mlperf::c::ConstructSUT, "Construct the system under test.");
  m.def("DestroySUT", &mlperf::c::DestroySUT,
        "Destroy the object created by ConstructSUT.");
  m.def("StartTest", &mlperf::c::StartTest,
        "Run tests on a SUT created by ConstructSUT().");
}

#endif  // PYTHON_BINDINGS_H
