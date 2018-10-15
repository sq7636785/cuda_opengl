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
    return r.origin + (t) * glm::normalize(r.direction);
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
    q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
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
        return glm::length(r.origin - intersectionPoint);
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

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotdiretion = glm::dot(rt.origin, rt.direction);
    float radicand = vDotdiretion * vDotdiretion - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
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

    return glm::length(r.origin - intersectionPoint);

}


__host__ __device__
float meshIntersectionTest(Geometry mesh, Triangle* tris, Ray r, glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float tMin = FLT_MAX;

    
    int nearestTriIndex = -1;
    glm::vec3 baryPosition(0.0f);
    glm::vec3 minBaryPosition(0.0f);
    
    bool isHit;
    for (int i = mesh.startIndex; i < mesh.endIndex; ++i) {
        if (glm::intersectRayTriangle(r.origin, r.direction,
            tris[i].vertices[0], tris[i].vertices[1], tris[i].vertices[2],
            baryPosition)) {
            // Only consider triangls in the ray direction
            // no triangels back!
            if (baryPosition.z > 0.f && baryPosition.z < tMin) {
                tMin = baryPosition.z;
                minBaryPosition = baryPosition;
                nearestTriIndex = i;
            }
        }
    }
    if (nearestTriIndex == -1) {
        return -1;
    }
    Triangle nearestIntersectTri = tris[nearestTriIndex];
    normal = nearestIntersectTri.normals[0] * (1.0f - minBaryPosition.x - minBaryPosition.y) +
        nearestIntersectTri.normals[1] * minBaryPosition.x +
        nearestIntersectTri.normals[2] * minBaryPosition.y;

    normal = glm::normalize(normal);

    return tMin;
}



#define MAX_BVH_INTERIOR_LEVEL 64

__host__ __device__
bool intersectBVH(const Ray& ray, ShadeableIntersection* isect, int& hitTriIdx, const LinearBVHNode* bvhNodes, const Triangle* primitives) {
    if (!bvhNodes) {
        return false;
    }

    bool hit = false;
    glm::vec3 invDir(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);
    int dirIsNeg[3] = { invDir.x < 0.0f, invDir.y < 0.0f, invDir.z < 0.0f };

    // Follow ray through BVH nodes to find primitive intersections
    int toVisitOffeset = 0;
    int currentNodeIndex = 0;
    int nodeToVisit[MAX_BVH_INTERIOR_LEVEL];

    while (true) {
        const LinearBVHNode *node = &bvhNodes[currentNodeIndex];

        //check ray against BVH node
        //ray is intersect with the BVH root node

        float tmpT;
        bool nodeIsect = (currentNodeIndex == 0) ? node->bounds.Intersect(ray, &tmpT) : true;

        if (node->bounds.IntersectP(ray, invDir, dirIsNeg)) {
            if (node->nPrimitives > 0) {
                //leaf node
                for (int i = 0; i < node->nPrimitives; ++i) {
                    ShadeableIntersection tmpIsect;
                    if (primitives[node->primitivesOffset + i].Intersect(ray, &tmpIsect)) {
                        hit = true;
                        if (isect->t == -1.0f) {
                            hitTriIdx = primitives[node->primitivesOffset + i].index;
                            (*isect) = tmpIsect;
                        } else {
                            if (tmpIsect.t < isect->t) {
                                (*isect) = tmpIsect;
                                hitTriIdx = primitives[node->primitivesOffset + i].index;
                            }
                        }
                    }
                }
                if (toVisitOffeset == 0) {
                    break;
                }
                currentNodeIndex = nodeToVisit[--toVisitOffeset];
            } else {
                //interior node

                // ----------- Depth control ---------------
                // if toVisitOffset reaches maximum
                // we don't want add more index to nodesToVisit Array
                // we just give up this interior node and handle previous nodes instead 
                if (toVisitOffeset == MAX_BVH_INTERIOR_LEVEL) {
                    currentNodeIndex = nodeToVisit[--toVisitOffeset];
                    continue;
                }

                //the travel order
                if (dirIsNeg[node->axis]) {
                    nodeToVisit[toVisitOffeset++] = currentNodeIndex + 1;
                    currentNodeIndex = node->secondChildOffset;
                } else {
                    nodeToVisit[toVisitOffeset++] = node->secondChildOffset;
                    currentNodeIndex = currentNodeIndex + 1;
                }
            }
        } else {
            //hit nothing
            if (toVisitOffeset == 0) {
                break;
            } else {
                currentNodeIndex = nodeToVisit[--toVisitOffeset];
            }
        }
    }
    return hit;
}


#endif // !INTERSECTION_H