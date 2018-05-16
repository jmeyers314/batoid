#include "rayVector.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportRayVector(py::module& m) {
        py::class_<RayVector>(m, "RayVector")
            .def(py::init<>())
            .def(py::init<RayVector>())
            .def(py::init<std::vector<Ray>>())
            .def("__repr__", &RayVector::repr)
            .def("amplitude", &RayVector::amplitude)
            .def("sumAmplitude", &RayVector::sumAmplitude)
            .def("phase", &RayVector::phase)
            .def("propagatedToTime", &RayVector::propagatedToTime)
            .def("propagateInPlace", &RayVector::propagateInPlace)
            .def("trimVignetted", &RayVector::trimVignetted)
            .def("trimVignettedInPlace", &RayVector::trimVignettedInPlace)
            .def(py::pickle(
                [](const RayVector& rv) {  // __getstate__
                    return py::make_tuple(rv.rays, rv.wavelength);
                },
                [](py::tuple t) {  // __setstate__
                    return RayVector(
                        t[0].cast<std::vector<Ray>>(),
                        t[1].cast<double>()
                    );
                }
            ))
            .def("__hash__", [](const RayVector& rv) {
                auto result = py::hash(py::make_tuple("RayVector", rv.wavelength));
                for (const auto& r : rv.rays) {
                    for (int i=0; i<3; i++) {
                        result = 1000003*result ^ py::hash(py::float_(r.p0[i]));
                        result = 1000003*result ^ py::hash(py::float_(r.v[i]));
                    }
                    result = 1000003*result ^ py::hash(py::make_tuple(r.t0, r.wavelength, r.isVignetted, r.failed));
                }
                result = (result == -1) ? -2 : result;
                return result;
            })
            .def_property_readonly(
                "x",
                [](RayVector& rv) -> py::array_t<double> {
                    return {{rv.size()},
                        {sizeof(Ray)},
                        &rv.rays[0].p0[0],
                        py::cast(rv)};
                }
            )
            .def_property_readonly(
                "y",
                [](RayVector& rv) -> py::array_t<double> {
                    return {{rv.size()},
                        {sizeof(Ray)},
                        &rv.rays[0].p0[1],
                        py::cast(rv)};
                }
            )
            .def_property_readonly(
                "z",
                [](RayVector& rv) -> py::array_t<double> {
                    return {{rv.size()},
                        {sizeof(Ray)},
                        &rv.rays[0].p0[2],
                        py::cast(rv)};
                }
            )
            .def_property_readonly(
                "vx",
                [](RayVector& rv) -> py::array_t<double> {
                    return {{rv.size()},
                        {sizeof(Ray)},
                        &rv.rays[0].v[0],
                        py::cast(rv)};
                }
            )
            .def_property_readonly(
                "vy",
                [](RayVector& rv) -> py::array_t<double> {
                    return {{rv.size()},
                        {sizeof(Ray)},
                        &rv.rays[0].v[1],
                        py::cast(rv)};
                }
            )
            .def_property_readonly(
                "vz",
                [](RayVector& rv) -> py::array_t<double> {
                    return {{rv.size()},
                        {sizeof(Ray)},
                        &rv.rays[0].v[2],
                        py::cast(rv)};
                }
            )
            .def_property_readonly(
                "t0",
                [](RayVector& rv) -> py::array_t<double> {
                    return {{rv.size()},
                        {sizeof(Ray)},
                        &rv.rays[0].t0,
                        py::cast(rv)};
                }
            )
            .def_property_readonly(
                "wavelength",
                [](RayVector& rv) -> py::array_t<double> {
                    return {{rv.size()},
                        {sizeof(Ray)},
                        &rv.rays[0].wavelength,
                        py::cast(rv)};
                }
            )
            .def_property_readonly(
                "isVignetted",
                [](RayVector& rv) -> py::array_t<bool> {
                    return {{rv.size()},
                        {sizeof(Ray)},
                        &rv.rays[0].isVignetted,
                        py::cast(rv)};
                }
            )
            .def_property_readonly(
                "failed",
                [](RayVector& rv) -> py::array_t<bool> {
                    return {{rv.size()},
                        {sizeof(Ray)},
                        &rv.rays[0].failed,
                        py::cast(rv)};
                }
            )
            .def_property_readonly(
                "v",
                [](RayVector& rv) -> py::array_t<double> {
                    return {{rv.size(), 3ul},
                        {sizeof(Ray), sizeof(double)},
                        &rv.rays[0].v[0],
                        py::cast(rv)};
                }
            )
            .def_property_readonly(
                "p0",
                [](RayVector& rv) -> py::array_t<double> {
                    return {{rv.size(), 3ul},
                        {sizeof(Ray), sizeof(double)},
                        &rv.rays[0].p0[0],
                        py::cast(rv)};
                }
            )
            .def_property_readonly(
                "k",
                [](const RayVector& rv) {
                    std::vector<Vector3d> result;
                    result.reserve(rv.size());
                    for (int i=0; i < rv.size(); i++) {
                        result.push_back(rv.rays[i].k());
                    }
                    return py::array_t<double>(
                        {rv.size(), 3ul},
                        {3*sizeof(double), sizeof(double)},
                        &result[0].data()[0]
                    );
                }
            )
            .def_property_readonly(
                "kx",
                [](const RayVector& rv) {
                    std::vector<double> result;
                    result.reserve(rv.size());
                    for (int i=0; i < rv.size(); i++) {
                        result.push_back(rv.rays[i].k()[0]);
                    }
                    return py::array_t<double>(
                        {rv.size()},
                        {sizeof(double)},
                        result.data()
                    );
                }
            )
            .def_property_readonly(
                "ky",
                [](const RayVector& rv) {
                    std::vector<double> result;
                    result.reserve(rv.size());
                    for (int i=0; i < rv.size(); i++) {
                        result.push_back(rv.rays[i].k()[1]);
                    }
                    return py::array_t<double>(
                        {rv.size()},
                        {sizeof(double)},
                        result.data()
                    );
                }
            )
            .def_property_readonly(
                "kz",
                [](const RayVector& rv) {
                    std::vector<double> result;
                    result.reserve(rv.size());
                    for (int i=0; i < rv.size(); i++) {
                        result.push_back(rv.rays[i].k()[2]);
                    }
                    return py::array_t<double>(
                        {rv.size()},
                        {sizeof(double)},
                        result.data()
                    );
                }
            )
            .def_property_readonly(
                "omega",
                [](const RayVector& rv) {
                    std::vector<double> result;
                    result.reserve(rv.size());
                    for (int i=0; i < rv.size(); i++) {
                        result.push_back(rv.rays[i].omega());
                    }
                    return py::array_t<double>(
                        {rv.size()},
                        {sizeof(double)},
                        &result[0]
                    );
                }
            )

            .def("__getitem__",
                [](RayVector &rv, typename std::vector<Ray>::size_type i) -> Ray& {
                    if (i >= rv.size())
                        throw py::index_error();
                    return rv.rays[i];
                },
                py::return_value_policy::reference_internal
            )
            .def("__iter__",
                [](RayVector &rv) {
                    return py::make_iterator<
                        py::return_value_policy::reference_internal,
                        typename std::vector<Ray>::iterator,
                        typename std::vector<Ray>::iterator,
                        Ray&>(rv.rays.begin(), rv.rays.end());
                },
                py::keep_alive<0, 1>()
            )
            .def("append", [](RayVector& rv, const Ray& r) { rv.rays.push_back(r); })
            .def("__len__", &RayVector::size)
            .def("__eq__", [](const RayVector& lhs, const RayVector& rhs) { return lhs.rays == rhs.rays; }, py::is_operator())
            .def("__ne__", [](const RayVector& lhs, const RayVector& rhs) { return lhs.rays != rhs.rays; }, py::is_operator());
    }
}
