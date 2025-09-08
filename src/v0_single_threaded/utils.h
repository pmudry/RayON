#pragma once

#include <memory>
#include <vector>
#include <cmath>
#include <iostream>
#include <limits>

using namespace std;

const double inf = numeric_limits<double>::infinity();
const double PI = 3.1415926535897932385;

#include "color.h"
#include "ray.h"
#include "vec3.h"
#include "interval.h"
#include "hittable.h"
#include "sphere.h"
#include "rnd_gen.h"
#include "camera.h"
