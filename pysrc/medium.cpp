#include "medium.h"
#include "table.h"
#include <memory>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace jtrace {
    void pyExportMedium(py::module& m) {
        py::class_<Medium, std::shared_ptr<Medium>>(m, "Medium")
            .def("getN", &Medium::getN);

        py::class_<ConstMedium, std::shared_ptr<ConstMedium>, Medium>(m, "ConstMedium")
            .def(py::init<double>());

        py::class_<TableMedium, std::shared_ptr<TableMedium>, Medium>(m, "TableMedium")
            .def(py::init<std::shared_ptr<Table<double,double>>>());
    }
}
