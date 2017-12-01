#include "medium.h"
#include "table.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportMedium(py::module& m) {
        py::class_<Medium, std::shared_ptr<Medium>>(m, "Medium")
            .def("getN", &Medium::getN);


        py::class_<ConstMedium, std::shared_ptr<ConstMedium>, Medium>(m, "ConstMedium")
            .def(py::init<double>())
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const ConstMedium& cm) { return py::make_tuple(cm.getN(0.0)); },
                [](py::tuple t) { return ConstMedium(t[0].cast<double>()); }
            ));


        py::class_<TableMedium, std::shared_ptr<TableMedium>, Medium>(m, "TableMedium")
            .def(py::init<std::shared_ptr<Table<double,double>>>())
            .def_property_readonly("table", &TableMedium::getTable)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const TableMedium& tm) { return py::make_tuple(*tm.getTable()); },
                [](py::tuple t) { return TableMedium(t[0].cast<std::shared_ptr<Table<double,double>>>()); }
            ));


        py::class_<SellmeierMedium, std::shared_ptr<SellmeierMedium>, Medium>(m, "SellmeierMedium")
            .def(py::init<double,double,double,double,double,double>())
            .def(py::init<std::array<double,6>>())
            .def_property_readonly("coefs", &SellmeierMedium::getCoefs)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const SellmeierMedium& sm) { return sm.getCoefs(); },
                [](std::array<double,6> coefs) {
                    return SellmeierMedium(coefs);
                }
            ));


        py::class_<Air, std::shared_ptr<Air>, Medium>(m, "Air")
            .def(py::init<double,double,double>(),
                 "init",
                 "pressure"_a=69.328, "temperature"_a=293.15, "h2o_pressure"_a=1.067)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::pickle(
                [](const Air& a) {
                    return py::make_tuple(
                        a.getPressure(),
                        a.getTemperature(),
                        a.getH2OPressure()
                    );
                },
                [](py::tuple t) {
                    return Air(t[0].cast<double>(), t[1].cast<double>(), t[2].cast<double>());
                }
            ));


    }
}
