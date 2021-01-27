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


        py::class_<TableMedium, std::shared_ptr<TableMedium>, Medium>(m, "CPPTableMedium")
            .def(py::init(
                [](
                    size_t w,
                    size_t n,
                    size_t size
                ){
                    return new TableMedium(
                        reinterpret_cast<double*>(w),
                        reinterpret_cast<double*>(n),
                        size
                    );
                }
            ));


        py::class_<SellmeierMedium, std::shared_ptr<SellmeierMedium>, Medium>(m, "CPPSellmeierMedium")
            .def(py::init<double,double,double,double,double,double>());


        py::class_<SumitaMedium, std::shared_ptr<SumitaMedium>, Medium>(m, "CPPSumitaMedium")
            .def(py::init<double,double,double,double,double,double>());


        py::class_<Air, std::shared_ptr<Air>, Medium>(m, "CPPAir")
            .def(py::init<double,double,double>());
    }
}
