#include "quadric.h"
#include <memory>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportQuadric(py::module& m) {
        py::class_<Quadric, std::shared_ptr<Quadric>, Surface>(m, "CPPQuadric")
            .def(py::init<double,double>(), "init", "R"_a, "conic"_a);
    }
}
