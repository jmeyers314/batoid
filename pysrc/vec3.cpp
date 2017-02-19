#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "vec3.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace jtrace {
    void pyExportVec3(py::module& m) {
        py::class_<Vec3>(m, "Vec3")
            .def(py::init<double,double,double>(), "init", "x"_a, "y"_a, "z"_a)
            .def(py::init<>())
            .def("MagnitudeSquared", &Vec3::MagnitudeSquared)
            .def("Magnitude", &Vec3::Magnitude)
            .def("UnitVec3", &Vec3::UnitVec3)
            .def("__repr__", &Vec3::repr)
            .def_readonly("x", &Vec3::x)
            .def_readonly("y", &Vec3::y)
            .def_readonly("z", &Vec3::z)
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def(py::self - py::self)
            .def(py::self -= py::self)
            .def(py::self * float())
            .def(py::self *= float())
            .def(py::self / float())
            .def(py::self /= float())
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(-py::self);

        py::class_<Rot3>(m, "Rot3", py::buffer_protocol())
            .def_buffer([](Rot3& r) -> py::buffer_info {
                return py::buffer_info(
                    r.data.data(),
                    sizeof(double),
                    py::format_descriptor<double>::format(),
                    2,
                    {3, 3},
                    {sizeof(double) * 3, sizeof(double)}
                );
            })
            .def(py::init<>())
            .def(py::init<std::array<double,9>>())
            .def("__repr__", &Rot3::repr)
            .def(py::self * float())
            .def(py::self *= float())
            .def(py::self / float())
            .def(py::self /= float())
            .def("determinant", &Rot3::determinant);

        m.def("DotProduct", &DotProduct);
        m.def("CrossProduct", &CrossProduct);
        m.def("RotVec", &RotVec);
        m.def("UnRotVec", &UnRotVec);
    }
}
