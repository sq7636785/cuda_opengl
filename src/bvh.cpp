#include <algorithm>
#include "bvh.h"


int g_maxPrimsInNode;

BVHBuildNode* recursiveBuild(std::vector<BVHPrimitiveInfo>& primitiveInfo,
    int start, int end, int& totalNodes,
    std::vector<Triangle> &orderedPrims,
    std::vector<Triangle> &primitives) {
    
    BVHBuildNode *node = new BVHBuildNode();
    totalNodes++;
    Bounds3f bounds = primitiveInfo[start].bounds;

    for (int i = start; i < end; ++i) {
        Union(bounds, primitiveInfo[i].bounds);
    }

    int nPrimitives = end - start;
    if (nPrimitives == 1) {
        //creat leaf
        int firstPrimOffest = orderedPrims.size();
        for (int i = start; i < end; ++i) {
            int primIdx = primitiveInfo[i].primitiveNumber;
            orderedPrims.push_back(primitives[primIdx]);
        }
        node->InitLeaf(firstPrimOffest, nPrimitives, bounds);
        return node;
    } else {
        //compute bound of primitive centroids, choose split dimension dim
        Bounds3f centroidBounds = Bounds3f(primitiveInfo[start].centroid);
        for (int i = start; i < end; ++i) {
            centroidBounds = Union(centroidBounds, primitiveInfo[i].centroid);
        }
        int dim = centroidBounds.MaximumExtent();

        float mid = static_cast<float>(start)+static_cast<float>(end)* 0.5f;
        if (centroidBounds.max[dim] == centroidBounds.min[dim]) {
            //creat leaf BVHBuildNode, the max axis equal, if means the bounds is a point.
            int firstPrimOffest = orderedPrims.size();
            for (int i = start; i < end; ++i) {
                int primIdx = primitiveInfo[i].primitiveNumber;
                orderedPrims.push_back(primitives[primIdx]);
            }
            node->InitLeaf(firstPrimOffest, nPrimitives, bounds);
            return node;
        } else {
            // ------------------------------------------
            // Partition primitives using approximate SAH
            // if nPrimitives is smaller than 4,
            // we partition them into equally sized subsets
            // ------------------------------------------

            //if (nPrimitives <= 4) {
                //partitation primitives into equally sized subsets
                std::nth_element(&primitiveInfo[start], &primitiveInfo[static_cast<int>(mid)], &primitiveInfo[end - 1] + 1,
                    [dim](const BVHPrimitiveInfo& a, const BVHPrimitiveInfo& b) {
                    return a.centroid[dim] < b.centroid[dim];
                });
            //}
        }
        node->InitInterior(dim,
            recursiveBuild(primitiveInfo, start, static_cast<int>(mid), totalNodes, orderedPrims, primitives),
            recursiveBuild(primitiveInfo, static_cast<int>(mid), end, totalNodes, orderedPrims, primitives));

    }
    return node;
}



int flattenBVHTree(BVHBuildNode *node, int *offset, LinerBVHNode *bvhNodes) {
    LinerBVHNode *linearNode = &bvhNodes[*offset];
    linearNode->bounds = node->bounds;
    int myOffset = (*offset)++;
    if (node->nPrimitives > 0) {
        linearNode->primitivesOffeset = node->firstPrimOffset;
        linearNode->nPrimitives = node->nPrimitives;
    } else {
        //create interior flattened BVH node
        linearNode->axis = node->splitAxis;
        linearNode->nPrimitives = 0;
        flattenBVHTree(node->children[0], offset, bvhNodes);
        linearNode->secondChildOffset = flattenBVHTree(node->children[1], offset, bvhNodes);
    }
    return myOffset;
}


void deleteBuildNode(BVHBuildNode *root) {
    if (root->children[0] == nullptr && root->children[1] == nullptr) {
        delete root;
        return;
    }

    deleteBuildNode(root->children[0]);
    deleteBuildNode(root->children[1]);

    delete root;
    return;
}


LinerBVHNode* ConstructBVHAccel(int& totalNodes, std::vector<Triangle>& primitives, int maxPrimsInNode) {
    g_maxPrimsInNode = glm::min(maxPrimsInNode, 255);

    size_t primitivesSize = primitives.size();
    if (primitivesSize == 0) {
        return nullptr;
    }

    //1. Initialize primitiveInfo array for primitives
    std::vector<BVHPrimitiveInfo> primitiveInfo(primitives.size());
    for (size_t i = 0; i < primitivesSize; ++i) {
        primitiveInfo[i] = BVHPrimitiveInfo(i, primitives[i].worldBounds());
    }

    //2.build BVH tree for primitives using primitiveInfo
    totalNodes = 0;
    std::vector<Triangle> orderedPrims(primitivesSize);
    
    BVHBuildNode *root;
    root = recursiveBuild(primitiveInfo, 0, primitivesSize, totalNodes, orderedPrims, primitives);
    primitives.swap(orderedPrims);

    //3.compute representation of depth-first traversal of BVH tree
    LinerBVHNode* bvhNodes = new LinerBVHNode[totalNodes];
    int offset = 0;
    flattenBVHTree(root, &offset, bvhNodes);

    //4.delete BVHBuildNode root
    deleteBuildNode(root);

    return bvhNodes;
}


