#include "medium.h"
#include "table.h"
#include <memory>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace jtrace {
    void pyExportMedium(py::module& m) {
        py::class_<Medium, std::shared_ptr<Medium>>(m, "Medium")
            .def("getN", &Medium::getN);

        py::class_<ConstMedium, std::shared_ptr<ConstMedium>, Medium>(m, "ConstMedium")
            .def(py::init<double>());

        py::class_<TableMedium, std::shared_ptr<TableMedium>, Medium>(m, "TableMedium")
            .def(py::init<std::shared_ptr<Table<double,double>>>());

        py::class_<SellmeierMedium, std::shared_ptr<SellmeierMedium>, Medium>(m, "SellmeierMedium")
            .def(py::init<double,double,double,double,double,double>());

        py::class_<Air, std::shared_ptr<Air>, Medium>(m, "Air")
            .def(py::init<double,double,double>(),
                 "init",
                 "p"_a=69.328, "t"_a=293.15, "w"_a=1.067);

    }
}
