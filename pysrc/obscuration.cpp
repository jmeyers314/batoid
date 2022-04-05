#include "obscuration.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportObscuration(py::module& m) {
        py::class_<ObscurationHandle, std::shared_ptr<ObscurationHandle>>(m, "CPPObscuration")
            .def("contains",
                [](const ObscurationHandle& h, size_t xarr, size_t yarr, size_t size, size_t outarr)
                {
                    double* xptr = reinterpret_cast<double*>(xarr);
                    double* yptr = reinterpret_cast<double*>(yarr);
                    bool* outptr = reinterpret_cast<bool*>(outarr);
                    auto obsc = h.getHostPtr();
                    for(int i=0; i<size; i++) {
                        outptr[i] = obsc->contains(xptr[i], yptr[i]);
                    }
                });


        py::class_<ObscCircleHandle, std::shared_ptr<ObscCircleHandle>, ObscurationHandle>(m, "CPPObscCircle")
            .def(py::init<double,double,double>());


        py::class_<ObscAnnulusHandle, std::shared_ptr<ObscAnnulusHandle>, ObscurationHandle>(m, "CPPObscAnnulus")
            .def(py::init<double,double,double,double>());


        py::class_<ObscRectangleHandle, std::shared_ptr<ObscRectangleHandle>, ObscurationHandle>(m, "CPPObscRectangle")
            .def(py::init<double,double,double,double,double>());


        py::class_<ObscRayHandle, std::shared_ptr<ObscRayHandle>, ObscurationHandle>(m, "CPPObscRay")
            .def(py::init<double,double,double,double>());


        py::class_<ObscPolygonHandle, std::shared_ptr<ObscPolygonHandle>, ObscurationHandle>(m, "CPPObscPolygon")
            .def(py::init(
                [](
                    size_t xp,
                    size_t yp,
                    size_t size
                ){
                    return new ObscPolygonHandle(
                        reinterpret_cast<double*>(xp),
                        reinterpret_cast<double*>(yp),
                        size
                    );
                }
            ));
            // .def(
            //     "containsGrid",
            //     [](
            //         const ObscPolygon& op,
            //         size_t x_ptr,
            //         size_t y_ptr,
            //         size_t out_ptr,
            //         size_t nx,
            //         size_t ny
            //     ){
            //         op.containsGrid(
            //             reinterpret_cast<const double*>(x_ptr),
            //             reinterpret_cast<const double*>(y_ptr),
            //             reinterpret_cast<bool*>(out_ptr),
            //             nx, ny
            //         );
            //     }
            // );


        py::class_<ObscNegationHandle, std::shared_ptr<ObscNegationHandle>, ObscurationHandle>(m, "CPPObscNegation")
            .def(py::init<ObscurationHandle*>());


        py::class_<ObscUnionHandle, std::shared_ptr<ObscUnionHandle>, ObscurationHandle>(m, "CPPObscUnion")
            .def(py::init(
                [](const std::vector<std::shared_ptr<ObscurationHandle>>& handles) {
                    const ObscurationHandle** _handles = new const ObscurationHandle*[handles.size()];
                    for (int i=0; i<handles.size(); i++) {
                        _handles[i] = handles[i].get();
                    }
                    return new ObscUnionHandle(_handles, handles.size());
                }
            ));

        py::class_<ObscIntersectionHandle, std::shared_ptr<ObscIntersectionHandle>, ObscurationHandle>(m, "CPPObscIntersection")
            .def(py::init(
                [](const std::vector<std::shared_ptr<ObscurationHandle>>& handles) {
                    const ObscurationHandle** _handles = new const ObscurationHandle*[handles.size()];
                    for (int i=0; i<handles.size(); i++) {
                        _handles[i] = handles[i].get();
                    }
                    return new ObscIntersectionHandle(_handles, handles.size());
                }
            ));
    }
}
