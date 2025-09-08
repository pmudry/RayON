#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "utils.h"
#include "hittable.h"

using namespace std;

class hittable_list : public hittable
{
    public: 
        vector<shared_ptr<hittable>> objects;

        hittable_list() {}
        hittable_list(shared_ptr<hittable> object) { add(object); }

        void clear() { objects.clear(); }

        void add(shared_ptr<hittable> object) { objects.push_back(object); }

        bool hit(const ray& r, double ray_tmin, double ray_tmax, hit_record& rec) const override {
            hit_record tmp;
            bool hitSomething = false;
            double closestSoFar = ray_tmax;

            for (int i = 0; i < objects.size(); i++)
            {
                // Check if the ray hits this object and if it's the closest hit so far
                if(objects[i]->hit(r, ray_tmin, closestSoFar, tmp)){
                    hitSomething = true;
                    closestSoFar = tmp.t;
                    rec = tmp;
                }
            }

            return hitSomething;
        }
};

#endif