#pragma once

#include "hittable.h"
#include "utils.h"

using namespace std;

class Hittable_list : public Hittable
{
 public:
   vector<shared_ptr<Hittable>> objects;

   Hittable_list() {}
   Hittable_list(shared_ptr<Hittable> object) { add(object); }

   void clear() { objects.clear(); }

   void add(shared_ptr<Hittable> object) { objects.push_back(object); }

   bool hit(const Ray &r, Interval ray_t, Hit_record &rec) const override
   {
      Hit_record tmp;
      bool hitSomething = false;
      double closestSoFar = ray_t.max;

      for (int i = 0; i < objects.size(); i++)
      {
         // Check if the ray hits this object and if it's the closest hit so far
         if (objects[i]->hit(r, Interval(ray_t.min, closestSoFar), tmp))
         {
            hitSomething = true;
            closestSoFar = tmp.t;
            rec = tmp;
         }
      }

      return hitSomething;
   }
};