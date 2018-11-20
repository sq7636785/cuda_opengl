#ifndef INTERSECTION_H
#define INTERSECTION_H

#include "glm/glm.hpp"
#include "glm/gtx/intersect.hpp"
#include "glm/gtc//matrix_transform.hpp"
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
float boxIntersectionTest(Geometry box, Ray r, glm::vec3 &intersectionPoint, glm::vec3 &normal, glm::vec2& uv, bool &outside) {
    
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

        glm::vec3 objspaceIntersection = getPointOnRay(q, tmin);
        glm::vec3 abs = glm::min(glm::abs(objspaceIntersection), 0.5f);
        glm::vec2 UV(0.0f);
        if (abs.x < abs.y && abs.x > abs.z) {
            UV = glm::vec2(objspaceIntersection.z + 0.5f, objspaceIntersection.y + 0.5f);
        } else if (abs.y > abs.x && abs.y > abs.z) {
            UV = glm::vec2(objspaceIntersection.x + 0.5f, objspaceIntersection.z + 0.5f);
        } else {
            UV = glm::vec2(objspaceIntersection.x + 0.5f, objspaceIntersection.y + 0.5f);
        }
        uv = UV;

        intersectionPoint = multiplyMV(box.transform, glm::vec4(objspaceIntersection, 1.0f));
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
float sphereIntersectionTest(Geometry sphere, Ray r, glm::vec3 &intersectionPoint, glm::vec3 &normal, glm::vec2 &uv, bool &outside) {
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

    //uv
    glm::vec3 p = glm::normalize(objspaceIntersection);
    float phi = atan2f(p.z, p.x);
    if (phi < 0.0f) {
        phi += TWO_PI;
    }
    float theta = glm::acos(p.y);
    uv = glm::vec2(1.0f - phi / TWO_PI, 1.0f - theta / PI);


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


//triangle

__host__ __device__
Bounds3f Triangle::worldBounds() {
    float minX = glm::min(vertices[0].x, glm::min(vertices[1].x, vertices[2].x));
    float minY = glm::min(vertices[0].y, glm::min(vertices[1].y, vertices[2].y));
    float minZ = glm::min(vertices[0].z, glm::min(vertices[1].z, vertices[2].z));

    float maxX = glm::max(vertices[0].x, glm::max(vertices[1].x, vertices[2].x));
    float maxY = glm::max(vertices[0].y, glm::max(vertices[1].y, vertices[2].y));
    float maxZ = glm::max(vertices[0].z, glm::max(vertices[1].z, vertices[2].z));


    if (minX == maxX) {
        minX -= 0.01f;
        maxX += 0.01f;
    }

    if (minY == maxY) {
        minY -= 0.01f;
        maxY += 0.01f;
    }

    if (minZ == maxZ) {
        minZ -= 0.01f;
        maxZ += 0.01f;
    }


    return Bounds3f(glm::vec3(minX, minY, minZ),
        glm::vec3(maxX, maxY, maxZ));
}


__host__ __device__ 
bool Triangle::Intersect(const Ray& r, ShadeableIntersection* isect) const {

    glm::vec3 baryPosition(0.f);

    if (glm::intersectRayTriangle(r.origin, r.direction,
        vertices[0], vertices[1], vertices[2],
        baryPosition)) {

        // Material ID should be set on the Geom level

        isect->t = baryPosition.z;

        isect->surfaceNormal = normals[0] * (1.0f - baryPosition.x - baryPosition.y) +
            normals[1] * baryPosition.x +
            normals[2] * baryPosition.y;

        isect->surfaceNormal = glm::normalize(isect->surfaceNormal);


        return true;
    }

    else {
        isect->t = -1.0f;
        return false;
    }

}

//curve

template<typename T>
__host__ __device__
T Lerp(float t, const T &p0, const T &p1) {
    return (1.0f - t) * p0 + t * p1;
}

__host__ __device__
glm::vec3 BlossomBezier(const glm::vec3 p[4], float u0, float u1, float u2) {
    glm::vec3 a[3] = {
        Lerp(u0, p[0], p[1]),
        Lerp(u0, p[1], p[2]),
        Lerp(u0, p[2], p[3])
    };
    glm::vec3 b[2] = {
        Lerp(u1, a[0], a[1]),
        Lerp(u1, a[1], a[2])
    };
    return Lerp(u2, b[0], b[1]);
}

__host__ __device__
inline void SubdivideBezier(const glm::vec3 cp[4], glm::vec3 cpSplit[7]) {
    cpSplit[0] = cp[0];
    cpSplit[1] = (cp[0] + cp[1]) / 2.0f;
    cpSplit[2] = (cp[0] + 2.0f * cp[1] + cp[2]) / 4.0f;
    cpSplit[3] = (cp[0] + 3.0f * cp[1] + 3.0f * cp[2] + cp[3]) / 8.0f;
    cpSplit[4] = (cp[1] + 2.0f * cp[2] + cp[3]) / 4.0f;
    cpSplit[5] = (cp[2] + cp[3]) / 2.0f;
    cpSplit[6] = cp[3];
}

__host__ __device__
static glm::vec3 EvalBezier(const glm::vec3 cp[4], float u, glm::vec3 *deriv = nullptr) {
    glm::vec3 cp1[3] = {
        Lerp(u, cp[0], cp[1]),
        Lerp(u, cp[1], cp[2]),
        Lerp(u, cp[2], cp[3])
    };
    glm::vec3 cp2[2] = {
        Lerp(u, cp1[0], cp1[1]),
        Lerp(u, cp1[1], cp1[2])
    };
    if (deriv) {
        glm::vec3 tmp = cp2[1] - cp2[0];
        float len = tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z;
        if (len > 0) {
            *deriv = 3.0f * (cp2[1] - cp2[0]);
        } else {
            *deriv = cp[3] - cp[1];
        }
    }
    return Lerp(u, cp2[0], cp2[1]);
}

__host__ __device__
inline uint64_t FloatToBits(double f) {
    uint64_t ui;
    memcpy(&ui, &f, sizeof(double));
    return ui;
}



template <typename T, typename U, typename V>
__host__ __device__
inline T Clamp(T val, U low, V high) {
    if (val < low)
        return low;
    else if (val > high)
        return high;
    else
        return val;
}


template<typename T>
__host__ __device__
inline Bounds3f Expand(const Bounds3f &b, T delta) {
    return Bounds3f(b.min - glm::vec3(delta, delta, delta),
        b.max + glm::vec3(delta, delta, delta));
}




Bounds3f Curve::objBound() const {
    glm::vec3 cpObj[4];
    cpObj[0] = BlossomBezier(common->controlPoint, uMin, uMin, uMin);
    cpObj[1] = BlossomBezier(common->controlPoint, uMin, uMin, uMax);
    cpObj[2] = BlossomBezier(common->controlPoint, uMin, uMax, uMax);
    cpObj[3] = BlossomBezier(common->controlPoint, uMax, uMax, uMax);

    Bounds3f b =
        Union(Bounds3f(cpObj[0], cpObj[1]), Bounds3f(cpObj[2], cpObj[3]));
    float width[2] = {
        Lerp(uMin, common->startWidth, common->endWidth),
        Lerp(uMin, common->startWidth, common->endWidth)
    };
    float mWidth = (width[0] > width[1]) ? (width[0]) : (width[1]);
    return Expand(b, mWidth * 0.5f);
}


__host__ __device__
void CoordinateSystem(const glm::vec3 &v1, glm::vec3 *v2,
    glm::vec3 *v3) {
    if (glm::abs(v1.x) > glm::abs(v1.y))
        *v2 = glm::vec3(-v1.z, 0, v1.x) / glm::sqrt(v1.x * v1.x + v1.z * v1.z);
    else
        *v2 = glm::vec3(0, v1.z, -v1.y) / glm::sqrt(v1.y * v1.y + v1.z * v1.z);
    *v3 = glm::cross(v1, *v2);
}


__host__ __device__
bool Curve::Intersect(const Ray& r, float *tHit, ShadeableIntersection *isect, glm::mat4 worldToObj, glm::mat4 objToWorld) {
    //** 要不要实现transform类
    glm::vec3 oErr, dErr;
    Ray ray;
    ray.origin = multiplyMV(worldToObj, glm::vec4(r.origin, 1.0f));
    ray.direction = multiplyMV(worldToObj, glm::vec4(r.direction, 0.0f));
    glm::vec3 cpObj[4];
    cpObj[0] = BlossomBezier(common->controlPoint, uMin, uMin, uMin);
    cpObj[1] = BlossomBezier(common->controlPoint, uMin, uMin, uMax);
    cpObj[2] = BlossomBezier(common->controlPoint, uMin, uMax, uMax);
    cpObj[3] = BlossomBezier(common->controlPoint, uMax, uMax, uMax);

    glm::vec3 dx = glm::cross(ray.direction, cpObj[3] - cpObj[0]);
    if (glm::dot(dx, dx) == 0.0f) {
        glm::vec3 dy;
        CoordinateSystem(ray.direction, &dx, &dy);
    }
    glm::mat4 objToRay = glm::lookAt(ray.origin, ray.origin + ray.direction, dx);
    glm::vec3 cp[4] = {
        multiplyMV(objToRay, glm::vec4(cpObj[0], 1.0f)),
        multiplyMV(objToRay, glm::vec4(cpObj[1], 1.0f)),
        multiplyMV(objToRay, glm::vec4(cpObj[2], 1.0f)),
        multiplyMV(objToRay, glm::vec4(cpObj[3], 1.0f))
    };

    float maxWidth = glm::max(Lerp(uMin, common->startWidth, common->endWidth),
        Lerp(uMax, common->startWidth, common->endWidth));
    if (glm::max(glm::max(cp[0].y, cp[1].y), glm::max(cp[2].y, cp[3].y)) +
        0.5f * maxWidth < 0 ||
        glm::min(glm::min(cp[0].y, cp[1].y), glm::min(cp[2].y, cp[3].y)) -
        0.5f * maxWidth > 0) {
        return false;
    }

    if (glm::max(glm::max(cp[0].x, cp[1].x), glm::max(cp[2].x, cp[3].x)) +
        0.5f * maxWidth < 0 ||
        glm::min(glm::min(cp[0].x, cp[1].x), glm::min(cp[2].x, cp[3].x)) -
        0.5f * maxWidth > 0) {
        return false;
    }

    float rayLength = glm::sqrt(glm::dot(ray.direction, ray.direction));
    float zMax = rayLength * FLT_MAX;
    if (glm::max(glm::max(cp[0].z, cp[1].z), glm::max(cp[2].z, cp[3].z)) +
        0.5f * maxWidth < 0 ||
        glm::min(glm::min(cp[0].z, cp[1].z), glm::min(cp[2].z, cp[3].z)) -
        0.5f * maxWidth > zMax) {
        return false;
    }

//     float L0 = 0;
//     for (int i = 0; i < 2; ++i)
//         L0 = std::max(
//         L0, std::max(
//         std::max(std::abs(cp[i].x - 2 * cp[i + 1].x + cp[i + 2].x),
//         std::abs(cp[i].y - 2 * cp[i + 1].y + cp[i + 2].y)),
//         std::abs(cp[i].z - 2 * cp[i + 1].z + cp[i + 2].z)));
// 
//     float eps =
//         std::max(common->startWidth, common->endWidth) * .05f;  // width / 20
//     auto Log2 = [](float v) -> int {
//         if (v < 1) return 0;
//         uint32_t bits = FloatToBits(v);
//         // https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
//         // (With an additional add so get round-to-nearest rather than
//         // round down.)
//         return (bits >> 23) - 127 + (bits & (1 << 22) ? 1 : 0);
//     };
    // Compute log base 4 by dividing log2 in half.
    int r0 = 0;//Log2(1.41421356237f * 6.f * L0 / (8.f * eps)) / 2;
    int maxDepth = Clamp(r0, 0, 10);

    //ReportValue(refinementLevel, maxDepth);
    return recursiveIntersect(ray, tHit, isect, cp, glm::inverse(objToRay), objToWorld, uMin, uMax, maxDepth);

}


__host__ __device__
bool Curve::recursiveIntersect(const Ray &ray, float *tHit, ShadeableIntersection *isect, const glm::vec3 cp[4], const glm::mat4 &rayToObject, glm::mat4 &objToWorld,
    float u0, float u1, int depth) const {
    float rayLength = glm::length(ray.direction);

//     if (depth > 0) {
//         glm::vec3 cpSplit[7];
//         SubdivideBezier(cp, cpSplit);
// 
//         bool hit = false;
//         float u[3] = {
//             u0,
//             (u0 + u1) / 2.0f,
//             u1
//         };
// 
//         const glm::vec3 *cps = cpSplit;
// 
//         for (int seg = 0; seg < 2; ++seg, cps += 3) {
//             float maxWidth =
//                 std::max(Lerp(u[seg], common->startWidth, common->endWidth),
//                 Lerp(u[seg + 1], common->startWidth, common->endWidth));
// 
//             if (std::max(std::max(cps[0].y, cps[1].y),
//                 std::max(cps[2].y, cps[3].y)) +
//                 0.5 * maxWidth < 0 ||
//                 std::min(std::min(cps[0].y, cps[1].y),
//                 std::min(cps[2].y, cps[3].y)) -
//                 0.5 * maxWidth > 0) {
//                 continue;
//             }
// 
//             if (std::max(std::max(cps[0].x, cps[1].x),
//                 std::max(cps[2].x, cps[3].x)) +
//                 0.5 * maxWidth < 0 ||
//                 std::min(std::min(cps[0].x, cps[1].x),
//                 std::min(cps[2].x, cps[3].x)) -
//                 0.5 * maxWidth > 0) {
//                 continue;
//             }
// 
//             float zMax = rayLength * FLT_MAX;
//             if (std::max(std::max(cps[0].z, cps[1].z),
//                 std::max(cps[2].z, cps[3].z)) +
//                 0.5 * maxWidth < 0 ||
//                 std::min(std::min(cps[0].z, cps[1].z),
//                 std::min(cps[2].z, cps[3].z)) -
//                 0.5 * maxWidth > zMax) {
//                 continue;
//             }
// 
//             hit |= recursiveIntersect(ray, tHit, isect, cps, rayToObject, objToWorld, u[seg], u[seg + 1], depth - 1);
//             if (hit && tHit) { return true; }
//         }
//         return hit;
//     } else {
        float edge = (cp[1].y - cp[0].y) * -cp[0].y + cp[0].x * (cp[0].x - cp[1].x);
        if (edge < 0) {
            return false;
        }

        edge = (cp[2].y - cp[3].y) * -cp[3].y + cp[3].x * (cp[3].x - cp[2].x);
        if (edge < 0) {
            return false;
        }

        glm::vec2 segmentDirection = glm::vec2(cp[3] - cp[0]);
        float denom = segmentDirection.x * segmentDirection.x + segmentDirection.y * segmentDirection.y;
        if (denom == 0.0f) {
            return false;
        }

        float w = glm::dot(glm::vec2(cp[0]), segmentDirection) / denom;
        float u = Clamp(Lerp(w, u0, u1), u0, u1);
        float hitWidth = Lerp(u, common->startWidth, common->endWidth);

        glm::vec3 nHit;

        glm::vec3 dpcdw;
        glm::vec3 pc = EvalBezier(cp, Clamp(w, 0, 1), &dpcdw);
        float ptCurveDist2 = pc.x * pc.x + pc.y * pc.y;
        if (ptCurveDist2 > hitWidth * hitWidth * .25) {
            return false;
        }
        float zMax = rayLength * FLT_MAX;
        if (pc.z < 0 || pc.z > zMax) {
            return false;
        }

        float ptCurveDist = std::sqrt(ptCurveDist2);
        float edgeFunc = dpcdw.x * -pc.y + pc.x * dpcdw.y;
        float v = (edgeFunc > 0) ? 0.5f + ptCurveDist / hitWidth
            : 0.5f - ptCurveDist / hitWidth;

        if (tHit != nullptr) {
            *tHit = pc.z / rayLength;

            glm::vec3 dpdu, dpdv;
            EvalBezier(common->controlPoint, u, &dpdu);


            // Compute curve $\dpdv$ for flat and cylinder curves
            glm::vec3 dpduPlane = multiplyMV(glm::inverse(rayToObject), glm::vec4(dpdu, 0.0f));

            glm::vec3 dpdvPlane =
                glm::normalize(glm::vec3(-dpduPlane.y, dpduPlane.x, 0)) * hitWidth;

            // Rotate _dpdvPlane_ to give cylindrical appearance
            float theta = Lerp(v, -90., 90.);
            glm::mat4 rot = glm::rotate(glm::mat4(), -theta, dpduPlane);
            dpdvPlane = multiplyMV(rot, glm::vec4(dpdvPlane, 0.0f));

            dpdv = multiplyMV(rayToObject, glm::vec4(dpdvPlane, 1.0f));

            isect->t = *tHit;
            glm::vec3 normal = glm::normalize(glm::cross(dpdu, dpdv));
            glm::vec3 intersectPoint = getPointOnRay(ray, pc.z);
            multiplyMV(objToWorld, glm::vec4(normal, 0.0f));
            multiplyMV(objToWorld, glm::vec4(intersectPoint, 1.0f));
            isect->surfaceNormal = normal;
            isect->intersectPoint = intersectPoint;
            //isect
            //这里主要是需要一个法向量， 看看法向是怎么是怎么算的。
            //normalize(cross(dpdu, dpdv))

        }
        return true;

    
}




#endif // !INTERSECTION_H