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
    return r.position + (t) * glm::normalize(r.diretion);
}

__host__ __device__
glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}



/**
* Test intersection between a ray and a transformed cube. Untransformed,
* the cube ranges from -0.5 to 0.5 in each axis and is centered at the position.
*
* @param intersectionPoint  Output parameter for point of intersection.
* @param normal             Output parameter for surface normal.
* @param outside            Output param for whether the ray came from outside.
* @return                   Ray parameter `t` value. -1 if no intersection.
*/

//transform the ray to local cooridinate
//calcute the intersect point in the box coordinate     Slabs method  https://blog.csdn.net/u012325397/article/details/50807880
//transform intersect point to world coordinate.
__host__ __device__
float boxIntersectionTest(Geometry box, Ray r, glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    
    Ray q;
    q.position = multiplyMV(box.inverseTransform, glm::vec4(r.position, 1.0f));
    q.diretion = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.diretion, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.diretion[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.position[xyz]) / qdxyz;
            float t2 = (+0.5f - q.position[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.transform, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.position - intersectionPoint);
    }
    return -1;
}

// CHECKITOUT
/**
* Test intersection between a ray and a transformed sphere. Untransformed,
* the sphere always has radius 0.5 and is centered at the position.
*
* @param intersectionPoint  Output parameter for point of intersection.
* @param normal             Output parameter for surface normal.
* @param outside            Output param for whether the ray came from outside.
* @return                   Ray parameter `t` value. -1 if no intersection.
*/

__host__ __device__
float sphereIntersectionTest(Geometry sphere, Ray r, glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.position, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.diretion, 0.0f)));

    Ray rt;
    rt.position = ro;
    rt.diretion = rd;

    float vDotdiretion = glm::dot(rt.position, rt.diretion);
    float radicand = vDotdiretion * vDotdiretion - (glm::dot(rt.position, rt.position) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotdiretion;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = glm::min(t1, t2);
        outside = true;
    } else {
        t = glm::max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.position - intersectionPoint);

}


__host__ __device__
float meshIntersectionTest(Geometry mesh, Triangle* tris, Ray r, glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float tMin = FLT_MAX;

    glm::vec3 hitPoint;
    glm::vec3 hitNormal;
    
    bool isHit;
    for (int i = mesh.startIndex; i < mesh.endIndex; ++i) {
        float t = tris[i].intersect(r, hitPoint, hitNormal, mesh.transform, mesh.inverseTransform, isHit);
        if (t > 0.0f) {
            if (t < tMin) {
                tMin = t;
                intersectionPoint = hitPoint;
                normal = hitNormal;
            }
        }
    }
    return tMin;
}

#endif // !INTERSECTION_H
