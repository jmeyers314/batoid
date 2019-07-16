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
            .def(py::init<std::vector<Ray>, double>())
            .def(py::init<std::vector<double>, std::vector<double>, std::vector<double>,
                          std::vector<double>, std::vector<double>, std::vector<double>,
                          std::vector<double>, std::vector<double>, std::vector<double>,
                          std::vector<bool>>())
            .def("__repr__", &RayVector::repr)
            .def("amplitude", &RayVector::amplitude)
            .def("sumAmplitude", &RayVector::sumAmplitude)
            .def("phase", &RayVector::phase)
            .def("positionAtTime",
                [](const RayVector& rv, double t){
                    std::vector<Vector3d> result(rv.positionAtTime(t));
                    return py::array_t<double>(
                        {result.size(), 3ul},
                        {3*sizeof(double), sizeof(double)},
                        &result[0].data()[0]
                    );
                }
            )
            .def("propagatedToTime", &RayVector::propagatedToTime)
            .def("propagateInPlace", &RayVector::propagateInPlace)
            .def("trimVignetted", &RayVector::trimVignetted, "", "minFlux"_a=0.0)
            .def("trimVignettedInPlace", &RayVector::trimVignettedInPlace, "", "minFlux"_a=0.0)
            // Note, monochromatic == True guarantees monochromaticity, but monochromatic == False,
            // doesn't guarantee polychromaticity.
            .def_property_readonly("monochromatic", [](const RayVector& rv){ return !std::isnan(rv.getWavelength()); })
            .def(py::pickle(
                [](const RayVector& rv) {  // __getstate__
                    return py::make_tuple(rv.getRays(), rv.getWavelength());
                },
                [](py::tuple t) {  // __setstate__
                    return RayVector(
                        t[0].cast<std::vector<Ray>>(),
                        t[1].cast<double>()
                    );
                }
            ))
            .def("__hash__", [](const RayVector& rv) {
                return py::hash(py::make_tuple(
                    "RayVector",
                    py::tuple(py::cast(rv.getRays())),
                    rv.getWavelength()
                ));
            })
            .def_property_readonly("x",
                [](RayVector& rv) -> py::array_t<double> {
                    return {{rv.size()},
                        {sizeof(Ray)},
                        &(rv.front()).r[0],
                        py::cast(rv)};
                }
            )
            .def_property_readonly("y",
                [](RayVector& rv) -> py::array_t<double> {
                    return {{rv.size()},
                        {sizeof(Ray)},
                        &(rv.front()).r[1],
                        py::cast(rv)};
                }
            )
            .def_property_readonly("z",
                [](RayVector& rv) -> py::array_t<double> {
                    return {{rv.size()},
                        {sizeof(Ray)},
                        &(rv.front()).r[2],
                        py::cast(rv)};
                }
            )
            .def_property_readonly("vx",
                [](RayVector& rv) -> py::array_t<double> {
                    return {{rv.size()},
                        {sizeof(Ray)},
                        &(rv.front()).v[0],
                        py::cast(rv)};
                }
            )
            .def_property_readonly("vy",
                [](RayVector& rv) -> py::array_t<double> {
                    return {{rv.size()},
                        {sizeof(Ray)},
                        &(rv.front()).v[1],
                        py::cast(rv)};
                }
            )
            .def_property_readonly("vz",
                [](RayVector& rv) -> py::array_t<double> {
                    return {{rv.size()},
                        {sizeof(Ray)},
                        &(rv.front()).v[2],
                        py::cast(rv)};
                }
            )
            .def_property_readonly("t",
                [](RayVector& rv) -> py::array_t<double> {
                    return {{rv.size()},
                        {sizeof(Ray)},
                        &(rv.front()).t,
                        py::cast(rv)};
                }
            )
            .def_property_readonly("wavelength",
                [](RayVector& rv) -> py::array_t<double> {
                    return {{rv.size()},
                        {sizeof(Ray)},
                        &(rv.front()).wavelength,
                        py::cast(rv)};
                }
            )
            .def_property_readonly("flux",
                [](RayVector& rv) -> py::array_t<double> {
                    return {{rv.size()},
                        {sizeof(Ray)},
                        &(rv.front()).flux,
                        py::cast(rv)};
                }
            )
            .def_property_readonly("vignetted",
                [](RayVector& rv) -> py::array_t<bool> {
                    return {{rv.size()},
                        {sizeof(Ray)},
                        &(rv.front()).vignetted,
                        py::cast(rv)};
                }
            )
            .def_property_readonly("failed",
                [](RayVector& rv) -> py::array_t<bool> {
                    return {{rv.size()},
                        {sizeof(Ray)},
                        &(rv.front()).failed,
                        py::cast(rv)};
                }
            )
            .def_property_readonly("v",
                [](RayVector& rv) -> py::array_t<double> {
                    return {{rv.size(), 3ul},
                        {sizeof(Ray), sizeof(double)},
                        &(rv.front()).v[0],
                        py::cast(rv)};
                }
            )
            .def_property_readonly("r",
                [](RayVector& rv) -> py::array_t<double> {
                    return {{rv.size(), 3ul},
                        {sizeof(Ray), sizeof(double)},
                        &(rv.front()).r[0],
                        py::cast(rv)};
                }
            )
            .def_property_readonly("k",
                [](const RayVector& rv) {
                    std::vector<Vector3d> result;
                    result.reserve(rv.size());
                    for (int i=0; i < rv.size(); i++) {
                        result.push_back(rv[i].k());
                    }
                    return py::array_t<double>(
                        {rv.size(), 3ul},
                        {3*sizeof(double), sizeof(double)},
                        &result[0].data()[0]
                    );
                }
            )
            .def_property_readonly("kx",
                [](const RayVector& rv) {
                    std::vector<double> result;
                    result.reserve(rv.size());
                    for (int i=0; i < rv.size(); i++) {
                        result.push_back(rv[i].k()[0]);
                    }
                    return py::array_t<double>(
                        {rv.size()},
                        {sizeof(double)},
                        result.data()
                    );
                }
            )
            .def_property_readonly("ky",
                [](const RayVector& rv) {
                    std::vector<double> result;
                    result.reserve(rv.size());
                    for (int i=0; i < rv.size(); i++) {
                        result.push_back(rv[i].k()[1]);
                    }
                    return py::array_t<double>(
                        {rv.size()},
                        {sizeof(double)},
                        result.data()
                    );
                }
            )
            .def_property_readonly("kz",
                [](const RayVector& rv) {
                    std::vector<double> result;
                    result.reserve(rv.size());
                    for (int i=0; i < rv.size(); i++) {
                        result.push_back(rv[i].k()[2]);
                    }
                    return py::array_t<double>(
                        {rv.size()},
                        {sizeof(double)},
                        result.data()
                    );
                }
            )
            .def_property_readonly("omega",
                [](const RayVector& rv) {
                    std::vector<double> result;
                    result.reserve(rv.size());
                    for (int i=0; i < rv.size(); i++) {
                        result.push_back(rv[i].omega());
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
                    return rv[i];
                },
                py::return_value_policy::reference_internal
            )
            .def("__iter__",
                [](RayVector &rv) {
                    return py::make_iterator<
                        py::return_value_policy::reference_internal,
                        typename std::vector<Ray>::iterator,
                        typename std::vector<Ray>::iterator,
                        Ray&>(rv.begin(), rv.end());
                },
                py::keep_alive<0, 1>()
            )
            .def("append", [](RayVector& rv, const Ray& r) { rv.push_back(r); })
            .def("__len__", &RayVector::size)
            .def("__eq__", [](const RayVector& lhs, const RayVector& rhs) { return lhs == rhs; }, py::is_operator())
            .def("__ne__", [](const RayVector& lhs, const RayVector& rhs) { return lhs != rhs; }, py::is_operator());
        m.def("concatenateRayVectors", &concatenateRayVectors);
    }
}
