#ifndef INTERSECTION_H
#define INTERSECTION_H

#include "glm/glm.hpp"
#include "glm/gtx/intersect.hpp"
#include "data_structure.h"
#include "utilities.h"


//Handy-dandy hash function that provides seeds for random number generation.

__host__ __device__ 
inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}



__host__ __device__
glm::vec3 getPointOnRay(Ray r, float t) {
    return r.position + (t - 0.0001f) * glm::normalize(r.diretion);
}

__host__ __device__
glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

__host__ __device__
float boxIntersectionTest(Geometry box, Ray r, glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    
}




#endif // !INTERSECTION_H
