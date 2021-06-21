#include "obscuration.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportObscuration(py::module& m) {
        py::class_<Obscuration, std::shared_ptr<Obscuration>>(m, "CPPObscuration")
            .def("contains", py::vectorize(&Obscuration::contains));


        py::class_<ObscCircle, std::shared_ptr<ObscCircle>, Obscuration>(m, "CPPObscCircle")
            .def(py::init<double,double,double>());


        py::class_<ObscAnnulus, std::shared_ptr<ObscAnnulus>, Obscuration>(m, "CPPObscAnnulus")
            .def(py::init<double,double,double,double>());


        py::class_<ObscRectangle, std::shared_ptr<ObscRectangle>, Obscuration>(m, "CPPObscRectangle")
            .def(py::init<double,double,double,double,double>());


        py::class_<ObscRay, std::shared_ptr<ObscRay>, Obscuration>(m, "CPPObscRay")
            .def(py::init<double,double,double,double>());


        py::class_<ObscPolygon, std::shared_ptr<ObscPolygon>, Obscuration>(m, "CPPObscPolygon")
            .def(py::init(
                [](
                    size_t xp,
                    size_t yp,
                    size_t size
                ){
                    return new ObscPolygon(
                        reinterpret_cast<double*>(xp),
                        reinterpret_cast<double*>(yp),
                        size
                    );
                }
            ))
            .def(
                "containsGrid",
                [](
                    const ObscPolygon& op,
                    size_t x_ptr,
                    size_t y_ptr,
                    size_t out_ptr,
                    size_t nx,
                    size_t ny
                ){
                    op.containsGrid(
                        reinterpret_cast<const double*>(x_ptr),
                        reinterpret_cast<const double*>(y_ptr),
                        reinterpret_cast<bool*>(out_ptr),
                        nx, ny
                    );
                }
            );


        py::class_<ObscNegation, std::shared_ptr<ObscNegation>, Obscuration>(m, "CPPObscNegation")
            .def(py::init<Obscuration*>());


        py::class_<ObscUnion, std::shared_ptr<ObscUnion>, Obscuration>(m, "CPPObscUnion")
            .def(py::init(
                [](const std::vector<std::shared_ptr<Obscuration>>& obscs) {
                    const Obscuration** _obscs = new const Obscuration*[obscs.size()];
                    for (int i=0; i<obscs.size(); i++) {
                        _obscs[i] = obscs[i].get();
                    }
                    return new ObscUnion(_obscs, obscs.size());
                }
            ));

        py::class_<ObscIntersection, std::shared_ptr<ObscIntersection>, Obscuration>(m, "CPPObscIntersection")
            .def(py::init(
                [](const std::vector<std::shared_ptr<Obscuration>>& obscs) {
                    const Obscuration** _obscs = new const Obscuration*[obscs.size()];
                    for (int i=0; i<obscs.size(); i++) {
                        _obscs[i] = obscs[i].get();
                    }
                    return new ObscIntersection(_obscs, obscs.size());
                }
            ));
    }
}
