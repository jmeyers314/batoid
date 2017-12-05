#include "table.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

namespace py = pybind11;

namespace batoid {
    void pyExportTable(py::module& m) {
        using TableDD = Table<double, double>;
        auto table = py::class_<TableDD, std::shared_ptr<TableDD>>(m, "Table")
            .def(py::init<std::vector<double>,std::vector<double>,TableDD::interpolant>())
            .def_property_readonly("args", &TableDD::getArgs)
            .def_property_readonly("vals", &TableDD::getVals)
            .def_property_readonly("interp", &TableDD::getInterp)
            .def("__call__", &TableDD::lookup)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const TableDD& t) {
                    return py::make_tuple(t.getArgs(), t.getVals(), t.getInterp());
                },
                [](py::tuple t) {
                    return TableDD(
                        t[0].cast<std::vector<double>>(),
                        t[1].cast<std::vector<double>>(),
                        t[2].cast<TableDD::interpolant>());
                }
            ))
            .def("__hash__", [](const TableDD& t) {
                auto result = py::hash(py::make_tuple("Table", t.getInterp()));
                for (const auto& arg : t.getArgs())
                    result ^= py::hash(py::float_(arg));
                for (const auto& val : t.getVals())
                    result ^= py::hash(py::float_(val));
                return result;
            })
            .def("__repr__", &TableDD::repr);

        py::enum_<TableDD::interpolant>(table, "Interpolant")
            .value("linear", TableDD::interpolant::linear)
            .value("floor", TableDD::interpolant::floor)
            .value("ceil", TableDD::interpolant::ceil)
            .value("nearest", TableDD::interpolant::nearest);
    }
}
