#ifndef batoid_transformation_h
#define batoid_transformation_h

#include <array>
#include <memory>
#include <sstream>
#include "surface.h"
#include "intersection.h"
#include "ray.h"
#include "vec3.h"

namespace batoid {
    struct Intersection;
    class Surface;
    class Transformation : public Surface {
    public:
        Transformation(std::shared_ptr<const Surface>, double dx, double dy, double dz);
        Transformation(std::shared_ptr<const Surface>, const Vec3& dr);
        Transformation(std::shared_ptr<const Surface>, const Rot3& r3);
        virtual double sag(double, double) const;
        virtual Vec3 normal(double, double) const;
        using Surface::intersect;
        virtual Intersection intersect(const Ray&) const;
        std::string repr() const;
        Vec3 getDr() const { return dr; }
        double getDx() const { return dr.x; }
        double getDy() const { return dr.y; }
        double getDz() const { return dr.z; }
        Rot3 getR() const { return rot; }
    private:
        const std::shared_ptr<const Surface> transformee;
        const Vec3 dr;
        const Rot3 rot;
    };

    inline std::ostream& operator<<(std::ostream& os, const Transformation& t);

    const Rot3 ident_rot {{{1,0,0,
                            0,1,0,
                            0,0,1}}};
}

#endif
