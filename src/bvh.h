#ifndef BVH_H
#define BVH_H

#include "data_structure.h"

// Forward declarations of structs used by our BVH tree
struct BVHPrimitiveInfo {
    int         primitiveNumber;
    Bounds3f    bounds;
    glm::vec3   centroid;

    BVHPrimitiveInfo() {}
    BVHPrimitiveInfo(size_t primitiveNumber_, const Bounds3f& bounds_) 
        :primitiveNumber(primitiveNumber_), bounds(bounds_), 
         centroid(0.5f * bounds_.min + 0.5f * bounds_.max) { }
};

struct BVHBuildNode {
    Bounds3f        bounds;
    BVHBuildNode*   children[2];
    int             splitAxis;
    int             firstPrimOffset;
    int             nPrimitives;

    void InitLeaf(int first, int n, const Bounds3f& b) {
        firstPrimOffset = first;
        nPrimitives = n;
        bounds = b;
        children[0] = children[1] = nullptr;
    }

    void InitInterior(int axis, BVHBuildNode *c0, BVHBuildNode *c1) {
        children[0] = c0;
        children[1] = c1;
        splitAxis = axis;
        bounds = Union(c0->bounds, c1->bounds);
        nPrimitives = 0;
    }
};

struct LinearBVHNode {
    Bounds3f bounds;
    union  {
        int primitivesOffeset;  // leaf
        int secondChildOffset;  // interior
    };

    unsigned short  nPrimitives;     // 0 -> interior node, 16 bytes
    unsigned char   axis;           // interior node: xyz, 8 bytes
    unsigned char   pad[1];         // ensure 32 byte total size
};

struct BucketInfo {
    int count = 0;
    Bounds3f bounds;
};


LinearBVHNode*   ConstructBVHAccel(int &totalNodes, std::vector<Triangle>& primitives, int maxPrimsInNode = 1);
void            DeconstructBVHAccel(LinearBVHNode* bvhNodes);



#endif