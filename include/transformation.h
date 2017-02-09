#ifndef __jem_transformation__h
#define __jem_transformation__h

#include "jtrace.h"
#include <array>
#include <memory>

namespace jtrace {
    struct Intersection;
    class Surface;
    class Transformation : public Surface {
    public:
        Transformation(std::shared_ptr<const Surface>, double dx, double dy, double dz);
        Transformation(std::shared_ptr<const Surface>, const Vec3& dr);
        Transformation(std::shared_ptr<const Surface>, std::array<std::array<double, 3>, 3> r3);
        virtual double operator()(double, double) const;
        virtual Vec3 normal(double, double) const;
        virtual Intersection intersect(const Ray&) const;
        std::string repr() const;
        Vec3 getDr() const { return dr; }
        double getDx() const { return dr.x; }
        double getDy() const { return dr.y; }
        double getDz() const { return dr.z; }
        std::array<std::array<double, 3>, 3> getR() const { return rot3; }
    private:
        const std::shared_ptr<const Surface> transformee;
        const Vec3 dr;
        const std::array<std::array<double, 3>, 3> rot3;
    };

    inline std::ostream& operator<<(std::ostream& os, const Transformation& t);

    constexpr std::array<std::array<double, 3>, 3> ident_rot3 {{{{1,0,0}},{{0,1,0}},{{0,0,1}}}};
}

#endif
