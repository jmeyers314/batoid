#include "medium.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportMedium(py::module& m) {
        py::class_<Medium, std::shared_ptr<Medium>>(m, "CPPMedium")
            .def("getN", py::vectorize(&Medium::getN));


        py::class_<ConstMedium, std::shared_ptr<ConstMedium>, Medium>(m, "CPPConstMedium")
            .def(py::init<double>());


        // py::class_<TableMedium, std::shared_ptr<TableMedium>, Medium>(m, "CPPTableMedium")
        //     .def(py::init<std::shared_ptr<Table<double,double>>>())
        //     .def_property_readonly("table", &TableMedium::getTable)
        //     .def(py::self == py::self)
        //     .def(py::self != py::self)
        //     .def(py::pickle(
        //         [](const TableMedium& tm) { return py::make_tuple(*tm.getTable()); },
        //         [](py::tuple t) { return TableMedium(t[0].cast<std::shared_ptr<Table<double,double>>>()); }
        //     ))
        //     .def("__hash__", [](const TableMedium& tm) {
        //         return py::hash(py::make_tuple(
        //             "CPPTableMedium",
        //             *tm.getTable()
        //         ));
        //     });


        py::class_<SellmeierMedium, std::shared_ptr<SellmeierMedium>, Medium>(m, "CPPSellmeierMedium")
            .def(py::init<double,double,double,double,double,double>());


        py::class_<SumitaMedium, std::shared_ptr<SumitaMedium>, Medium>(m, "CPPSumitaMedium")
            .def(py::init<double,double,double,double,double,double>());


        py::class_<Air, std::shared_ptr<Air>, Medium>(m, "CPPAir")
            .def(py::init<double,double,double>());
    }
}
