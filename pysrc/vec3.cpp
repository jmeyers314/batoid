#include <stdexcept>
#include "vec3.h"
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportVec3(py::module& m) {
        PYBIND11_NUMPY_DTYPE(Vec3, x, y, z);
        py::class_<Vec3>(m, "Vec3", R"pbdoc(
          Simple python 3-vector

          Parameters
          ----------
          x
            x-coordinate
          y
            y-coordinate
          z
            z-coordinate

          Notes
          -----
          `x`, `y`, and `z` may also all be left blank, in which case the 0-vector is returned.

          Examples
          --------
          >>> v = batoid.Vec3(1, 2, 3.4)
          >>> print(v)
          Vec3(1, 2, 3.4)

          >>> v2 = batoid.Vec3()
          >>> print(v2)
          Vec3(0, 0, 0)
        )pbdoc")
            .def(py::init<double,double,double>(), "init", "x"_a, "y"_a, "z"_a)
            .def(py::init<>())
            .def(py::init<std::array<double,3>>())
            .def("MagnitudeSquared", &Vec3::MagnitudeSquared, "Return square of vector magnitude.")
            .def("Magnitude", &Vec3::Magnitude, "Return vector magnitude.")
            .def("UnitVec3", &Vec3::UnitVec3, "Return unit vector pointing in same direction.")
            .def("__repr__", &Vec3::repr)
            .def_readonly("x", &Vec3::x, "x-coordinate of vector")
            .def_readonly("y", &Vec3::y, "y-coordinate of vector")
            .def_readonly("z", &Vec3::z, "z-coordinate of vector")
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def(py::self - py::self)
            .def(py::self -= py::self)
            .def(py::self * double())
            .def(py::self *= double())
            .def(py::self / double())
            .def(py::self /= double())
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(-py::self)
            .def(py::pickle(
                [](const Vec3& r) { // __getstate__
                    return py::make_tuple(r.x, r.y, r.z);
                },
                [](py::tuple t) { // __setstate__
                    if (t.size() != 3)
                        throw std::runtime_error("Invalid state!");
                    return Vec3(t[0].cast<double>(), t[1].cast<double>(), t[2].cast<double>());
                }
            ));

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
            .def(py::init<std::array<double,3>>())
            .def(py::init<double,double,double>())
            .def(py::self * py::self)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def("__repr__", &Rot3::repr)
            .def("determinant", &Rot3::determinant)
            .def_property_readonly("euler", &Rot3::getEuler)
            .def(py::pickle(
                [](const Rot3& r) { // __getstate__
                    return py::make_tuple(r.data);
                },
                [](py::tuple t) {
                    if (t.size() != 1)
                        throw std::runtime_error("Invalid state!");
                    return Rot3(t[0].cast<std::array<double,9>>());
                }
            ));

        m.def("DotProduct", &DotProduct, R"pbdoc(
          Compute the dot-product of two Vec3 objects.

          Parameters
          ----------
          v1 : Vec3
            First vector
          v2 : Vec3
            Second vector

          Returns
          -------
          float
            The dot product.

          Notes
          -----
          The dot product is defined as the sum of the component-wise products of two vectors.  It
          is useful for computing the magnitude of a vector, or the angle between two vectors as

          .. math::
            v1 \dot v2 = \cos(\theta)

          where :math:`\theta` is the angle in between the two vectors.

          Examples
          --------
          >>> v1 = batoid.Vec3(0, 1, 0)
          >>> v2 = batoid.Vec3(0, 0, 1)
          >>> print(batoid.DotProduct(v1, v2))
          0.0
          >>> v1 = batoid.Vec3(0, 1, 2)
          >>> v2 = batoid.Vec3(1, 2, 3)
          >>> assert batoid.DotProduct(v1, v2) == v1.x*v2.x + v1.y*v2.y + v1.z*v2.z

        )pbdoc");

        m.def("CrossProduct", &CrossProduct, R"pbdoc(
          Compute the cross-product of two Vec3 objects.

          Parameters
          ----------
          v1 : Vec3
            First vector
          v2 : Vec3
            Second vector

          Returns
          -------
          Vec3
            The vector cross-product v1 x v2.

          Notes
          -----
          The vector cross-product is useful for computing a vector that is perpendicular to both
          `v1` and `v2`.  The magnitude of the cross-product is equal to

          .. math::
            |v1 \cross v2| = |v1|  |v2| \sin(\theta)

          where :math:`\theta` is the angle in between `v1` and `v2`.  batoid Vec3 objects obey the
          right-hand-rule, where Vec3(1, 0, 0) x Vec3(0, 1, 0) = Vec3(0, 0, 1), and cyclic
          permutations thereof.

          Examples
          --------
          >>> v1 = batoid.Vec3(0, 1, 2)
          >>> v2 = batoid.Vec3(1, 2, 3)
          >>> batoid.CrossProduct(v1, v2)
          Vec3(-1, 2, -1)

        )pbdoc");
        m.def("RotVec", &RotVec);
        m.def("UnRotVec", &UnRotVec);
    }
}
