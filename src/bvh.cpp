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
        bounds = Union(bounds, primitiveInfo[i].bounds);
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

        int mid = static_cast<int>((static_cast<float>(start)+static_cast<float>(end))* 0.5f);
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

            if (nPrimitives <= 2) {
                //partitation primitives into equally sized subsets

                std::nth_element(&primitiveInfo[start], &primitiveInfo[static_cast<int>(mid)], &primitiveInfo[end - 1] + 1,
                    [dim](const BVHPrimitiveInfo& a, const BVHPrimitiveInfo& b) {
                    return a.centroid[dim] < b.centroid[dim];
                });
            } else {
                //sah
                const int nBuckets = 12;
                BucketInfo buckets[nBuckets];

                for (int i = start; i < end; ++i) {
                    int b = nBuckets * centroidBounds.Offset(primitiveInfo[i].centroid)[dim];
                    if (b == nBuckets) {
                        b = nBuckets - 1;
                    }
                    buckets[b].count++;
                    buckets[b].bounds = Union(buckets[b].bounds, primitiveInfo[i].bounds);
                }

                //calculate cost
                float cost[nBuckets - 1];
                for (int i = 0; i < nBuckets - 1; ++i) {
                    Bounds3f b0;
                    Bounds3f b1;
                    int count0 = 0;
                    int count1 = 0;
                    for (int j = 0; j <= i; ++j) {
                        b0 = Union(b0, buckets[j].bounds);
                        count0 += buckets[j].count;
                    }
                    for (int k = i + 1; k < nBuckets; ++k) {
                        b1 = Union(b1, buckets[k].bounds);
                        count1 += buckets[k].count;
                    }
                    cost[i] = 1.0f + (b0.SurfaceArea() * count0 + b1.SurfaceArea() * count1) / bounds.SurfaceArea();
                }

                //find min cost
                float minCost = FLT_MAX;
                int minID = -1;
                for (int i = 0; i < nBuckets - 1; ++i) {
                    if (cost[i] < minCost) {
                        minCost = cost[i];
                        minID = i;
                    }
                }

                float leafCost = static_cast<float>(nPrimitives);
                //partition according to sah min cost
                if (nPrimitives > g_maxPrimsInNode || leafCost > minCost) {
                    BVHPrimitiveInfo *pmid = std::partition(
                        &primitiveInfo[start],
                        &primitiveInfo[end - 1] + 1,
                        [=](const BVHPrimitiveInfo& pi) {
                        int b = nBuckets * centroidBounds.Offset(pi.centroid)[dim];
                        if (b == nBuckets) {
                            b = nBuckets - 1;
                        }
                        return b <= minID;
                    });
                    mid = pmid - &primitiveInfo[0];
                } else {
                    //leaf node;
                    int firstOffest = orderedPrims.size();
                    for (int i = start; i < end; ++i) {
                        int primitiveIdx = primitiveInfo[i].primitiveNumber;
                        orderedPrims.push_back(primitives[primitiveIdx]);
                    }
                    node->InitLeaf(firstOffest, nPrimitives, bounds);
                    return node;
                }

            }

            node->InitInterior(dim,
                recursiveBuild(primitiveInfo, start, static_cast<int>(mid), totalNodes, orderedPrims, primitives),
                recursiveBuild(primitiveInfo, static_cast<int>(mid), end, totalNodes, orderedPrims, primitives));
        }
    }
    return node;
}



int flattenBVHTree(BVHBuildNode *node, int *offset, LinearBVHNode *bvhNodes) {
    LinearBVHNode *linearNode = &bvhNodes[*offset];
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


LinearBVHNode* ConstructBVHAccel(int& totalNodes, std::vector<Triangle>& primitives, int maxPrimsInNode) {
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
    std::vector<Triangle> orderedPrims;
    orderedPrims.reserve(totalNodes);

    
    BVHBuildNode *root;
    root = recursiveBuild(primitiveInfo, 0, primitivesSize, totalNodes, orderedPrims, primitives);
    primitives.swap(orderedPrims);

    //3.compute representation of depth-first traversal of BVH tree
    LinearBVHNode* bvhNodes = new LinearBVHNode[totalNodes];
    int offset = 0;
    flattenBVHTree(root, &offset, bvhNodes);

    //4.delete BVHBuildNode root
    deleteBuildNode(root);

    return bvhNodes;
}


void DeconstructBVHAccel(LinearBVHNode* bvhNodes) {
    delete[] bvhNodes;
}