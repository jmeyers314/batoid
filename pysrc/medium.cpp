#include "medium.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportMedium(py::module& m) {
        py::class_<MediumHandle, std::shared_ptr<MediumHandle>>(m, "CPPMedium")
            .def("getN",
                [](const MediumHandle& h, size_t warr, size_t size, size_t outarr)
                {
                    double* wptr = reinterpret_cast<double*>(warr);
                    double* outptr = reinterpret_cast<double*>(outarr);
                    auto medium = h.getHostPtr();
                    for(int i=0; i<size; i++) {
                        outptr[i] = medium->getN(wptr[i]);
                    }
                });


        py::class_<ConstMediumHandle, std::shared_ptr<ConstMediumHandle>, MediumHandle>(m, "CPPConstMedium")
            .def(py::init<double>());


        py::class_<TableMediumHandle, std::shared_ptr<TableMediumHandle>, MediumHandle>(m, "CPPTableMedium")
            .def(py::init(
                [](
                    size_t w,
                    size_t n,
                    size_t size
                ){
                    return new TableMediumHandle(
                        reinterpret_cast<double*>(w),
                        reinterpret_cast<double*>(n),
                        size
                    );
                }
            ));


        py::class_<SellmeierMediumHandle, std::shared_ptr<SellmeierMediumHandle>, MediumHandle>(m, "CPPSellmeierMedium")
            .def(py::init<double,double,double,double,double,double>());


        py::class_<SumitaMediumHandle, std::shared_ptr<SumitaMediumHandle>, MediumHandle>(m, "CPPSumitaMedium")
            .def(py::init<double,double,double,double,double,double>());


        py::class_<AirHandle, std::shared_ptr<AirHandle>, MediumHandle>(m, "CPPAir")
            .def(py::init<double,double,double>());
    }
}
