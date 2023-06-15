#ifndef PERLIN_H
#define PERLIN_H

#include "torrey.h"
#include <vector>
#include "image.cuh"
#include "pcg.cuh"

// Borrowed from RTOW Chapter 7
/*
inline double random_double() {
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0);
}

// Borrowed from RTOW Chapter 7
inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double();
}

// Borrowed from RTNW Chapter 3
inline int random_int(int min, int max) {
    // Returns a random integer in [min,max].
    return static_cast<int>(random_double(min, max+1));
}
*/

// Adapted from RTNW Chapter 5
class perlin {
    public:
        perlin() {
            pcg32_state rng = init_pcg32(0);
            ranfloat = new double[point_count];
            for (int i = 0; i < point_count; ++i) {
                ranfloat[i] = next_pcg32_real<Real>(rng);
            }

            perm_x = perlin_generate_perm();
            perm_y = perlin_generate_perm();
            perm_z = perlin_generate_perm();
        }

        ~perlin() {
            delete[] ranfloat;
            delete[] perm_x;
            delete[] perm_y;
            delete[] perm_z;
        }

        double noise(const Vector3& p) const {
            auto scaled = 64.0 * p;
            auto i = static_cast<int>(4*scaled.x) & 255;
            auto j = static_cast<int>(4*scaled.y) & 255;
            auto k = static_cast<int>(4*scaled.z) & 255;

            return ranfloat[perm_x[i] ^ perm_y[j] ^ perm_z[k]];
        }

    private:
        static const int point_count = 256;
        double* ranfloat;
        int* perm_x;
        int* perm_y;
        int* perm_z;

        static int* perlin_generate_perm() {
            auto p = new int[point_count];

            for (int i = 0; i < perlin::point_count; i++)
                p[i] = i;

            permute(p, point_count);

            return p;
        }

        static void permute(int* p, int n) {
            pcg32_state rng = init_pcg32(0);
            for (int i = n-1; i > 0; i--) {
                int target =  int(next_pcg32_real<Real>(rng))%i;;
                int tmp = p[i];
                p[i] = p[target];
                p[target] = tmp;
            }
        }
};

#endif 