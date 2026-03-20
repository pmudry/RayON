
/**
 * @class Hittable_list
 * @brief Represents a collection of hittable objects in a ray tracer.
 *
 * This class is a derived class of `Hittable` and is used to manage a list of
 * hittable objects. It provides functionality to add objects to the list,
 * clear the list, and determine if a ray intersects with any of the objects
 * in the list.
 *
 * @note The `hit` method checks for the closest intersection of a ray with
 *       the objects in the list and updates the hit record accordingly.
 */
#pragma once

#include "hittable.hpp"
// RndGen is available transitively via hittable.hpp → vec3.hpp → rnd_gen.hpp

using namespace std;

class Hittable_list : public Hittable
{
 public:
   vector<shared_ptr<Hittable>> objects;

   Hittable_list() {}
   Hittable_list(shared_ptr<Hittable> object) { add(object); }

   void clear() { objects.clear(); }
   void add(shared_ptr<Hittable> object) { objects.push_back(object); }
   bool empty() const { return objects.empty(); }

   bool hit(const Ray &r, Interval ray_t, Hit_record &rec) const override
   {
      Hit_record tmp;
      bool hitSomething = false;
      double closestSoFar = ray_t.max;

      for (int i = 0; i < (int)objects.size(); i++)
      {
         if (objects[i]->hit(r, Interval(ray_t.min, closestSoFar), tmp))
         {
            hitSomething  = true;
            closestSoFar  = tmp.t;
            rec           = tmp;
         }
      }

      return hitSomething;
   }

   /// Mixture PDF: average solid-angle PDF over all objects in the list.
   /// Used for sampling from a set of area lights.
   double pdf_value(const Point3 &origin, const Vec3 &direction) const override
   {
      if (objects.empty())
         return 0.0;
      double sum = 0.0;
      for (const auto &obj : objects)
         sum += obj->pdf_value(origin, direction);
      return sum / static_cast<double>(objects.size());
   }

   /// Sample a direction toward one of the objects, chosen uniformly at random.
   Vec3 random_direction(const Point3 &origin) const override
   {
      if (objects.empty())
         return Vec3(1, 0, 0);
      int idx = static_cast<int>(RndGen::random_double() * static_cast<double>(objects.size()));
      idx     = std::min(idx, static_cast<int>(objects.size()) - 1);
      return objects[idx]->random_direction(origin);
   }
};
