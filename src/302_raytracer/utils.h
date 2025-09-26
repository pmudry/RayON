#pragma once

#include <memory>
#include <vector>
#include <cmath>
#include <iostream>
#include <limits>

using namespace std;

const double inf = numeric_limits<double>::infinity();
const double PI = 3.1415926535897932385;

namespace utils{
    inline double degrees_to_radians(double degrees) { return degrees * M_PI / 180.0; }
};
