#include "sphere.h"
#include <memory>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportSphere(py::module& m) {
        py::class_<Sphere, std::shared_ptr<Sphere>, Surface>(m, "CPPSphere")
            .def(py::init<double>(), "init", "R"_a);
    }
}
