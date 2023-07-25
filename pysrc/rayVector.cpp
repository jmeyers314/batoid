#include "rayVector.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/complex.h>

namespace py = pybind11;

namespace batoid {
    void pyExportRayVector(py::module& m) {
        auto dvd = py::class_<DualView<double>>(m, "CPPDualViewDouble")
            .def(py::init(
                [](
                    size_t arr_ptr,
                    size_t size
                ){
                    return new DualView<double>(reinterpret_cast<double*>(arr_ptr), size);
                }
            ))
            .def("syncToHost", &DualView<double>::syncToHost)
            .def("syncToDevice", &DualView<double>::syncToDevice)
            .def_readonly("size", &DualView<double>::size)
            .def_readonly("ownsHostData", &DualView<double>::ownsHostData);

        auto dvb = py::class_<DualView<bool>>(m, "CPPDualViewBool")
            .def("syncToHost", &DualView<bool>::syncToHost)
            .def("syncToDevice", &DualView<bool>::syncToDevice)
            .def_readonly("size", &DualView<bool>::size)
            .def_readonly("ownsHostData", &DualView<bool>::ownsHostData);

        auto rv = py::class_<RayVector>(m, "CPPRayVector")
            .def(py::init(
                [](
                    size_t x_ptr,
                    size_t y_ptr,
                    size_t z_ptr,
                    size_t vx_ptr,
                    size_t vy_ptr,
                    size_t vz_ptr,
                    size_t t_ptr,
                    size_t w_ptr,
                    size_t f_ptr,
                    size_t vig_ptr,
                    size_t fail_ptr,
                    size_t size
                ){
                    return new RayVector(
                        reinterpret_cast<double*>(x_ptr),
                        reinterpret_cast<double*>(y_ptr),
                        reinterpret_cast<double*>(z_ptr),
                        reinterpret_cast<double*>(vx_ptr),
                        reinterpret_cast<double*>(vy_ptr),
                        reinterpret_cast<double*>(vz_ptr),
                        reinterpret_cast<double*>(t_ptr),
                        reinterpret_cast<double*>(w_ptr),
                        reinterpret_cast<double*>(f_ptr),
                        reinterpret_cast<bool*>(vig_ptr),
                        reinterpret_cast<bool*>(fail_ptr),
                        size
                    );
                }
            ))
            .def("positionAtTime",
                [](const RayVector& rv, double t,
                   size_t xout_ptr, size_t yout_ptr, size_t zout_ptr,
                   int max_threads
                ){
                    rv.positionAtTime(
                        t,
                        reinterpret_cast<double*>(xout_ptr),
                        reinterpret_cast<double*>(yout_ptr),
                        reinterpret_cast<double*>(zout_ptr),
                        max_threads
                    );
                }
            )
            .def("propagateInPlace", &RayVector::propagateInPlace)
            .def("phase",
                [](const RayVector& rv,
                   double x, double y, double z, double t,
                   size_t out_ptr, int max_threads
                ){
                    rv.phase(x, y, z, t, reinterpret_cast<double*>(out_ptr), max_threads);
                }
            )
            .def("amplitude",
                [](const RayVector& rv,
                   double x, double y, double z, double t,
                   size_t out_ptr, int max_threads
                ){
                    rv.amplitude(
                        x, y, z, t,
                        reinterpret_cast<std::complex<double>*>(out_ptr),
                        max_threads
                    );
                }
            )
            .def("sumAmplitude", &RayVector::sumAmplitude)
            .def(py::self == py::self)
            .def(py::self != py::self)

            // Expose dualviews so can access their syncToHost methods
            .def_readonly("x", &RayVector::x)
            .def_readonly("y", &RayVector::y)
            .def_readonly("z", &RayVector::z)
            .def_readonly("vx", &RayVector::vx)
            .def_readonly("vy", &RayVector::vy)
            .def_readonly("vz", &RayVector::vz)
            .def_readonly("t", &RayVector::t)
            .def_readonly("wavelength", &RayVector::wavelength)
            .def_readonly("flux", &RayVector::flux)
            .def_readonly("vignetted", &RayVector::vignetted)
            .def_readonly("failed", &RayVector::failed)
            ;
    }
}
