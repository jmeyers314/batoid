#include "quadric2.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportQuadric2(py::module& m) {
        py::class_<Quadric2, std::shared_ptr<Quadric2>, Surface2>(m, "CPPQuadric2")
            .def(py::init<double,double>(), "init");
    }
}
