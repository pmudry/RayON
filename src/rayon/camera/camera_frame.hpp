#pragma once

#include "data_structures/vec3.hpp"

struct CameraFrame
{
   int image_width = 0;
   int image_height = 0;
   int image_channels = 0;
   int samples_per_pixel = 1;
   int max_depth = 1;
   Point3 camera_center;
   Point3 pixel00_loc;
   Vec3 pixel_delta_u;
   Vec3 pixel_delta_v;
   Vec3 u;
   Vec3 v;
   Vec3 w;
};
