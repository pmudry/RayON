#pragma once

#include "utils.h"
#include "hittable.h"

/**
 * Rectangle class that can serve as both a surface and an area light
 */
class rectangle : public hittable
{
public:
    rectangle(const point3 &corner, const vec3 &u, const vec3 &v) 
        : corner(corner), u(u), v(v), is_light(false), light_color(color(1,1,1)), light_intensity(1.0)
    {
        normal = unit_vector(cross(u, v));
        area = u.length() * v.length();
    }

    // Constructor for area light
    rectangle(const point3 &corner, const vec3 &u, const vec3 &v, const color &light_col, double intensity) 
        : corner(corner), u(u), v(v), is_light(true), light_color(light_col), light_intensity(intensity)
    {
        normal = unit_vector(cross(u, v));
        area = u.length() * v.length();
    }

    bool hit(const ray &r, interval ray_t, hit_record &rec) const override
    {
        // Calculate the plane equation: normal · (P - corner) = 0
        // For ray P = origin + t * direction
        // normal · (origin + t * direction - corner) = 0
        // Solve for t: t = normal · (corner - origin) / (normal · direction)
        
        double denom = dot(normal, r.direction());
        
        // If denominator is close to 0, ray is parallel to plane
        if (fabs(denom) < 1e-8)
            return false;
            
        double t = dot(normal, corner - r.origin()) / denom;
        
        if (!ray_t.surrounds(t))
            return false;
            
        // Find the intersection point
        point3 intersection = r.at(t);
        
        // Check if intersection is within rectangle bounds
        vec3 p = intersection - corner;
        
        // Project onto rectangle's coordinate system
        double alpha = dot(p, u) / dot(u, u);
        double beta = dot(p, v) / dot(v, v);
        
        // Check if point is inside rectangle (0 <= alpha <= 1, 0 <= beta <= 1)
        if (alpha < 0.0 || alpha > 1.0 || beta < 0.0 || beta > 1.0)
            return false;
            
        // We have a valid hit
        rec.t = t;
        rec.p = intersection;
        rec.set_face_normal(r, normal);
        
        // Mark as light if this is an area light
        if (is_light) {
            rec.isMirror = false; // Reuse existing field to mark as light
        }
        
        return true;
    }

    // Sample a random point on the rectangle for area lighting
    point3 sample_point() const
    {
        double alpha = RndGen::random_double();
        double beta = RndGen::random_double();
        return corner + alpha * u + beta * v;
    }

    // Get the area of the rectangle
    double get_area() const { return area; }
    
    // Check if this rectangle is a light source
    bool is_area_light() const { return is_light; }
    
    // Get light properties
    color get_light_color() const { return light_color; }
    double get_light_intensity() const { return light_intensity; }

private:
    point3 corner;      // One corner of the rectangle
    vec3 u, v;          // Two edges of the rectangle from corner
    vec3 normal;        // Normal vector to the rectangle
    double area;        // Area of the rectangle
    
    // Light properties
    bool is_light;
    color light_color;
    double light_intensity;
};