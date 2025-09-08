#pragma once

#include "utils.h"
#include "hittable.h"

class sphere : public hittable
{
public:
    sphere(const point3 &center, double radius) : center(center), radius(std::fmax(0, radius)) {}

    /**
     * @brief Calculates the intersection point of a ray with a sphere
     *
     * This function determines if and where a ray intersects with a sphere by solving
     * the quadratic equation formed by substituting the ray equation into the sphere equation.
     *
     * **Mathematical Background:**
     * - Sphere equation: `(C−P)⋅(C−P) = r²` where P is a point on the sphere
     * - Ray equation: `P = O + t*d` where O is origin, d is direction, t is distance
     * - We then solve for t by substituting the ray equation into the sphere equation
     *
     * @param center The center point of the sphere in 3D space
     * @param radius The radius of the sphere (must be positive)
     * @param r The ray to test for intersection
     *
     * @return The parameter t for the intersection point along the ray:
     *     - Returns `-1.0` if no intersection occurs (discriminant < 0)
     *     - Returns the farther intersection point when two intersections exist TODO: check if closer or farther, I think it's wrong
     *     - The actual intersection point can be computed as `r.origin() + t * r.direction()`
     *
     * @note When the discriminant is non-negative, this function returns the larger t value,
     *       corresponding to the exit point of the ray from the sphere
     */
    bool hit(const ray &r, interval ray_t, hit_record &rec) const override
    {
        vec3 oc = center - r.origin();
        auto a = r.direction().length_squared();
        auto h = dot(r.direction(), oc);
        auto c = oc.length_squared() - radius * radius;

        auto discriminant = h * h - a * c;
        if (discriminant < 0)
            return false;

        auto sqrtd = std::sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (h - sqrtd) / a;
        if(ray_t.surrounds(root) == false)
        {
            root = (h + sqrtd) / a;
            if (ray_t.surrounds(root) == false)
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        rec.normal = (rec.p - center) / radius;

        return true;
    }

private:
    point3 center;
    double radius;
};