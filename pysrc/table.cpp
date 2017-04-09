#include "table.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace jtrace {
    void pyExportTable(py::module& m) {
        using TableDD = Table<double, double>;
        auto table = py::class_<TableDD, std::shared_ptr<TableDD>>(m, "Table")
            .def(py::init<std::vector<double>,std::vector<double>,TableDD::interpolant>())
            .def_property_readonly("args", &TableDD::getArgs)
            .def_property_readonly("vals", &TableDD::getVals)
            .def_property_readonly("interp", &TableDD::getInterp)
            .def("__call__", &TableDD::lookup);

        py::enum_<TableDD::interpolant>(table, "Interpolant")
            .value("linear", TableDD::interpolant::linear)
            .value("floor", TableDD::interpolant::floor)
            .value("ceil", TableDD::interpolant::ceil)
            .value("nearest", TableDD::interpolant::nearest);
    }
}
