#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "data_structure.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "path_tracer.h"
#include "intersections.h"
#include "interactions.h"
#include "device_launch_parameters.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}



__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}


//Kernel that writes the image to the OpenGL PBO directly.
__global__
void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if (x < resolution.x && y < resolution.y) {
        int index = y * resolution.x + x;
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        float total = static_cast<float>(iter);
        color.x = glm::clamp(static_cast<int>(pix.x / total * 255.0), 0, 255);
        color.y = glm::clamp(static_cast<int>(pix.y / total * 255.0), 0, 255);
        color.z = glm::clamp(static_cast<int>(pix.z / total * 255.0), 0, 255);

        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}


static Scene*                   hst_scene = NULL;
static glm::vec3*               dev_image = NULL;
static Geometry*                dev_geometry = NULL;
static Material*                dev_material = NULL;
static PathSegment*             dev_paths = NULL;
static ShadeableIntersection*   dev_intersection = NULL;
static Triangle*                dev_tris = NULL;



#ifdef ENABLE_MESHWORLDBOUND
static int                      worldBoundsSize = 0;
static Bounds3f*                dev_worldBounds = NULL;
#endif

#ifdef ENABLE_BVH
static int                      bvhNodesSize = 0;
static LinearBVHNode*           dev_bvhNodes = NULL;
#endif

static Texture*                 dev_environmentMap = NULL;

static Texture*                 dev_textureMap = NULL;
static int                      textureMapSize = 0;

static Curve*                   dev_curves = NULL;


void pathTraceInit(Scene* scene) {
    hst_scene = scene;
    const Camera& cam = hst_scene->state.camera;
    const int pixelNum = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelNum * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelNum * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelNum * sizeof(PathSegment));

    cudaMalloc(&dev_geometry, hst_scene->geometrys.size() * sizeof(Geometry));
    cudaMemcpy(dev_geometry, hst_scene->geometrys.data(), hst_scene->geometrys.size() * sizeof(Geometry), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_material, hst_scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_material, hst_scene->materials.data(), hst_scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersection, pixelNum * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersection, 0, pixelNum * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_tris, hst_scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_tris, hst_scene->triangles.data(), hst_scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_curves, hst_scene->curves.size() * sizeof(Curve));
    cudaMemcpy(dev_curves, hst_scene->curves.data(), hst_scene->curves.size() * sizeof(Curve), cudaMemcpyHostToDevice);

#ifdef ENABLE_MESHWORLDBOUND
    worldBoundsSize = scene->worldBounds.size();
    if (worldBoundsSize > 0) {
        // World bounds of mesh
        cudaMalloc(&dev_worldBounds, worldBoundsSize * sizeof(Bounds3f));
        cudaMemcpy(dev_worldBounds, scene->worldBounds.data(), worldBoundsSize * sizeof(Bounds3f), cudaMemcpyHostToDevice);
    }
#endif

#ifdef ENABLE_BVH
    bvhNodesSize = hst_scene->bvhTotalNodes;
    if (bvhNodesSize > 0) {
        cudaMalloc(&dev_bvhNodes, bvhNodesSize * sizeof(LinearBVHNode));
        cudaMemcpy(dev_bvhNodes, hst_scene->bvhNodes, bvhNodesSize * sizeof(LinearBVHNode), cudaMemcpyHostToDevice);
    }
#endif

    int environmentMapSize = hst_scene->environmentMap.size();
    if (environmentMapSize > 0) {
        //move texture from cpu to gpu in Texture class
        int pixelNum = hst_scene->environmentMap[0].width * hst_scene->environmentMap[0].height * hst_scene->environmentMap[0].nComp;
        cudaMalloc(&(hst_scene->environmentMap[0].devData), sizeof(unsigned char) * pixelNum);
        cudaMemcpy(hst_scene->environmentMap[0].devData, hst_scene->environmentMap[0].hostData, pixelNum * sizeof(unsigned char), cudaMemcpyHostToDevice);
        
        cudaMalloc(&dev_environmentMap, sizeof(Texture));
        cudaMemcpy(dev_environmentMap, hst_scene->environmentMap.data(),sizeof(Texture), cudaMemcpyHostToDevice);
    }
    
    textureMapSize = hst_scene->textureMap.size();
    if (textureMapSize > 0) {
        for (int i = 0; i < textureMapSize; ++i) {
            int pixelNum = hst_scene->textureMap[i].height * hst_scene->textureMap[i].height * hst_scene->textureMap[i].nComp;
            cudaMalloc(&hst_scene->textureMap[i].devData, sizeof(unsigned char)* pixelNum);
            cudaMemcpy(hst_scene->textureMap[i].devData, hst_scene->textureMap[i].hostData, sizeof(unsigned char)* pixelNum, cudaMemcpyHostToDevice);
        }
        cudaMalloc(&dev_textureMap, sizeof(Texture)* textureMapSize);
        cudaMemcpy(dev_textureMap, hst_scene->textureMap.data(), textureMapSize * sizeof(Texture), cudaMemcpyHostToDevice);
    }

    checkCUDAError("pathTraceInit");
}


void pathTraceFree() {
    cudaFree(dev_geometry);
    cudaFree(dev_material);
    cudaFree(dev_image);
    cudaFree(dev_intersection);
    cudaFree(dev_paths);
    cudaFree(dev_tris);
    cudaFree(dev_curves);
#ifdef ENABLE_MESHWORLDBOUND
    if (worldBoundsSize > 0) {
        cudaFree(dev_worldBounds);
    }
#endif

#ifdef ENABLE_BVH
    if (bvhNodesSize > 0) {
        cudaFree(dev_bvhNodes);
    }
#endif

    if (dev_environmentMap != NULL) {
        
      //  cudaFree(hst_scene->environmentMap[0].devData);
        
        cudaFree(dev_environmentMap);
    }

    if (textureMapSize > 0) {
        cudaFree(dev_textureMap);
    }
    checkCUDAError("pathTraceFree");
}


/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__
void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = y * cam.resolution.x + x;

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, x, y);
        thrust::uniform_real_distribution<float> u01(0, 1);

        float xMov = u01(rng);
        float yMov = u01(rng);

        PathSegment &tmp = pathSegments[index];
        tmp.ray.origin = cam.position;
        tmp.remainingBounces = traceDepth;
        tmp.pixelIndex = index;
        tmp.color = glm::vec3(1.0, 1.0, 1.0);
        // TODO: implement antialiasing by jittering the ray
        tmp.ray.direction = glm::normalize(cam.view - (static_cast<float>(x)+xMov - cam.resolution.x * 0.5f) * cam.pixelLength.x * cam.right
            - (static_cast<float>(y)+yMov - cam.resolution.y * 0.5f) * cam.pixelLength.y * cam.up);

        if (cam.lenRadius > 0.0f) {
            float u1 = u01(rng);
            float u2 = u01(rng);
            glm::vec2 pLens = concentricSampleDisk(u1, u2) * cam.lenRadius;
            glm::vec3 pFocus = tmp.ray.origin + tmp.ray.direction * cam.focalDistance;
            tmp.ray.origin += pLens.x * cam.right + pLens.y * cam.up;
            tmp.ray.direction = glm::normalize(pFocus - tmp.ray.origin);
        }
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__
void computeIntersection(
    int num_paths, 
    PathSegment* pathSegments, 
    Geometry* geoms, 
    int geoms_size, 
    ShadeableIntersection *intersections,
    Triangle* tris,
    Curve*  curves
#ifdef ENABLE_MESHWORLDBOUND
    ,Bounds3f *worldBounds
#endif
#ifdef ENABLE_BVH
    ,LinearBVHNode* bvhNodes
#endif
    ) {
    
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < num_paths) {
        //final para
        float tMin = 10000.0f;
        glm::vec3 normal;
        glm::vec2 uv;
        int geomsId = -1;
        PathSegment &unit = pathSegments[index];
        ShadeableIntersection &si = intersections[index];
        //tmp para
        bool outside = true;
        glm::vec3 tmpIntersectPoint;
        glm::vec3 tmpNormal;
        glm::vec2 tmpUV;
        float t;

        int hitTriIdx = -1;
        for (int i = 0; i < geoms_size; ++i) {
            if (geoms[i].type == GeomType::SPHERE) {
                t = sphereIntersectionTest(geoms[i], unit.ray, tmpIntersectPoint, tmpNormal, tmpUV, outside);
            } else if (geoms[i].type == GeomType::CUBE) {
                t = boxIntersectionTest(geoms[i], unit.ray, tmpIntersectPoint, tmpNormal, tmpUV, outside);
            } else if (geoms[i].type == GeomType::MESH) {
#ifdef ENABLE_MESHWORLDBOUND
#ifdef ENABLE_BVH
                ShadeableIntersection tmpIsect;
                tmpIsect.t = FLT_MAX;
                if (intersectBVH(unit.ray, &tmpIsect, hitTriIdx, bvhNodes, tris)) {
                    if (hitTriIdx >= geoms[i].startIndex && hitTriIdx < geoms[i].endIndex) {
                        t = tmpIsect.t;
                        tmpNormal = tmpIsect.surfaceNormal;
                    } else {
                        t = -1.0f;
                    }
                } else {
                    t = -1.0f;
                }
#else
                float tmp_t;
                if (worldBounds[geoms[i].worldBoundIdx].Intersect(unit.ray, &tmp_t)) {
                    t = meshIntersectionTest(geoms[i], tris, unit.ray, tmpIntersectPoint, normal, outside);
                }
#endif
#else
                t = meshIntersectionTest(geoms[i], tris, unit.ray, intersectPoint, normal, outside);
#endif
            } else if (geoms[i].type == GeomType::CURVE) {
                ShadeableIntersection isect;
                isect.t = FLT_MAX;
                Geometry& geom = geoms[i];
                Curve& curve = curves[geom.curveId];
                
                glm::mat4 &trans = geom.transform;
                glm::mat4 &invtrans = geom.invTranspose;
                //if (curve.Intersect(unit.ray, &tmpIsect.t, &tmpIsect, trans, invtrans)) {}

                Ray r = unit.ray;
                CurveCommon *common = curve.common;
                float uMin = curve.uMin;
                float uMax = curve.uMax;


                glm::vec3 oErr, dErr;
                Ray ray;
                ray.origin = multiplyMV(trans, glm::vec4(r.origin, 1.0f));
                ray.direction = multiplyMV(trans, glm::vec4(r.direction, 0.0f));
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
                
                //zaizhe
                float maxWidth = glm::max(Lerp(uMin, common->startWidth, common->endWidth),
                    Lerp(uMax, common->startWidth, common->endWidth));
                if (glm::max(glm::max(cp[0].y, cp[1].y), glm::max(cp[2].y, cp[3].y)) +
                    0.5f * maxWidth < 0 ||
                    glm::min(glm::min(cp[0].y, cp[1].y), glm::min(cp[2].y, cp[3].y)) -
                    0.5f * maxWidth > 0) {
                    t = -1.0f;
                    continue;
                }

                if (glm::max(glm::max(cp[0].x, cp[1].x), glm::max(cp[2].x, cp[3].x)) +
                    0.5f * maxWidth < 0 ||
                    glm::min(glm::min(cp[0].x, cp[1].x), glm::min(cp[2].x, cp[3].x)) -
                    0.5f * maxWidth > 0) {
                    t = -1.0f;
                    continue;
                }
                /*
                float rayLength = glm::sqrt(glm::dot(ray.direction, ray.direction));
                float zMax = rayLength * FLT_MAX;
                if (glm::max(glm::max(cp[0].z, cp[1].z), glm::max(cp[2].z, cp[3].z)) +
                    0.5f * maxWidth < 0 ||
                    glm::min(glm::min(cp[0].z, cp[1].z), glm::min(cp[2].z, cp[3].z)) -
                    0.5f * maxWidth > zMax) {
                    t = -1.0f;
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


                //recur
                glm::mat4 rayToObj = glm::inverse(objToRay);
                float u0 = uMin;
                float u1 = uMax;
                //return recursiveIntersect(ray, tHit, isect, cp, glm::inverse(objToRay), objToWorld, uMin, uMax, maxDepth);
                float edge = (cp[1].y - cp[0].y) * -cp[0].y + cp[0].x * (cp[0].x - cp[1].x);
                if (edge < 0) {
                    t = -1.0f;
                    continue;
                }

                edge = (cp[2].y - cp[3].y) * -cp[3].y + cp[3].x * (cp[3].x - cp[2].x);
                if (edge < 0) {
                    t = -1.0f;
                    continue;
                }

                glm::vec2 segmentDirection = glm::vec2(cp[3] - cp[0]);
                float denom = segmentDirection.x * segmentDirection.x + segmentDirection.y * segmentDirection.y;
                if (denom == 0.0f) {
                    t = -1.0f;
                    continue;
                }

                float w = glm::dot(glm::vec2(cp[0]), segmentDirection) / denom;
                float u = Clamp(Lerp(w, u0, u1), u0, u1);
                float hitWidth = Lerp(u, common->startWidth, common->endWidth);

                glm::vec3 nHit;

                glm::vec3 dpcdw;
                glm::vec3 pc = EvalBezier(cp, Clamp(w, 0, 1), &dpcdw);
                float ptCurveDist2 = pc.x * pc.x + pc.y * pc.y;
                if (ptCurveDist2 > hitWidth * hitWidth * .25) {
                    t = -1.0f;
                    continue;
                }
                zMax = rayLength * FLT_MAX;
                if (pc.z < 0 || pc.z > zMax) {
                    t = -1.0f;
                    continue;
                }

                float ptCurveDist = std::sqrt(ptCurveDist2);
                float edgeFunc = dpcdw.x * -pc.y + pc.x * dpcdw.y;
                float v = (edgeFunc > 0) ? 0.5f + ptCurveDist / hitWidth
                    : 0.5f - ptCurveDist / hitWidth;

                if (t != 0.0) {
                    t = pc.z / rayLength;

                    glm::vec3 dpdu, dpdv;
                    EvalBezier(common->controlPoint, u, &dpdu);


                    // Compute curve $\dpdv$ for flat and cylinder curves
                    glm::vec3 dpduPlane = multiplyMV(objToRay, glm::vec4(dpdu, 0.0f));

                    glm::vec3 dpdvPlane =
                        glm::normalize(glm::vec3(-dpduPlane.y, dpduPlane.x, 0)) * hitWidth;

                    // Rotate _dpdvPlane_ to give cylindrical appearance
                    float theta = Lerp(v, -90., 90.);
                    glm::mat4 rot = glm::rotate(glm::mat4(), -theta, dpduPlane);
                    dpdvPlane = multiplyMV(rot, glm::vec4(dpdvPlane, 0.0f));

                    dpdv = multiplyMV(rayToObj, glm::vec4(dpdvPlane, 1.0f));

                    
                    glm::vec3 tnormal = glm::normalize(glm::cross(dpdu, dpdv));
                    glm::vec3 intersectPoint = getPointOnRay(ray, pc.z);
                    tmpNormal = multiplyMV(geom.invTranspose, glm::vec4(tnormal, 0.0f));
                    tmpIntersectPoint = multiplyMV(geom.inverseTransform, glm::vec4(intersectPoint, 1.0f));
                    
                    
                    //isect
                    //这里主要是需要一个法向量， 看看法向是怎么是怎么算的。
                    //normalize(cross(dpdu, dpdv))

                }
                */
                /*
#ifdef ENABLE_MESHWORLDBOUND
                float tmp_t;
                if (worldBounds[geoms[i].worldBoundIdx].Intersect(unit.ray, &tmp_t)) {
                    if (curves[geom.curveId].Intersect(unit.ray, &t, &tmpIsect, geom.transform, geom.invTranspose)) {
                        //就是拿到交点和normal
                        tmpNormal = tmpIsect.surfaceNormal;
                    }
                } else {
                    t = -1.0f;
                }
                
#else
                
                if (curves[geom.curveId].Intersect(unit.ray, &t, &tmpIsect, geom.transform, geom.invTranspose)) {
                    //就是拿到交点和normal
                    tmpNormal = tmpIsect.surfaceNormal;
                } else {
                    t = -1.0f;
                }
                */

            }
            //TODO:  more geometry type
            
            if (t < tMin && t > 0) {
                tMin = t;
                normal = tmpNormal;
                geomsId = i;
                uv = tmpUV;
            }
        }
        if (tMin > 0 && geomsId != -1) {
            si.materialId = geoms[geomsId].materialID;
            si.surfaceNormal = normal;
            si.uv = uv;
            si.t = tMin;
            si.hitGeomId = geomsId;
        } else {
            si.t = -1.0f;
        }
    }

}


// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.

__global__
void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* intersections,
    PathSegment* pathSegments,
    Material* materials) {

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < num_paths) {
        ShadeableIntersection &intersect = intersections[index];
        Material &m = materials[intersect.materialId];
        PathSegment &pathSegment = pathSegments[index];
        if (intersect.t > 0.0) {
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            if (m.emittance > 0) {
                pathSegment.color *= (m.emittance * m.color);
            } else {
                float lightTerm = glm::dot(intersect.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[index].color *= (m.color * lightTerm) * 0.3f + ((1.0f - intersect.t * 0.02f) * m.color) * 0.7f;
                pathSegments[index].color *= u01(rng); // apply some noise because why not
            }
        } else {
            pathSegment.color = glm::vec3(0.0f);
        }
    };
}


__global__
void shadeMaterial(
      int dp
    , int iter
    , int num_paths
    , ShadeableIntersection* intersections
    , PathSegment* pathSegments
    , Material* materials
    , Geometry* geoms
    , Triangle* tris
    , Texture* environmentMap
    , Texture* textureMap
#ifdef ENABLE_MESHWORLDBOUND
    , Bounds3f* worldBounds
#endif
#ifdef ENABLE_BVH
    , LinearBVHNode* bvhNodes
#endif
    ) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < num_paths) {
        ShadeableIntersection intersect = intersections[index];
        Material material = materials[intersect.materialId];
        PathSegment &pathSegment = pathSegments[index];

        if (pathSegment.remainingBounces > 0) {
            
            if (intersect.t > 0) {

                if (material.emittance > 0) {
                    pathSegment.color *= (material.emittance * material.color);
                    pathSegment.remainingBounces = 0;
                } else {

                    //这里参数的最后一个不能是0,不然会出bug
                    thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, dp);
                    glm::vec3 intersectPoint = getPointOnRay(pathSegment.ray, intersect.t);
                    scatterRay(
                        pathSegment
                      , intersectPoint
                      , intersect.surfaceNormal
                      , intersect.uv
                      , material
                      , geoms
                      , tris
                      , textureMap
#ifdef ENABLE_MESHWORLDBOUND
                      , worldBounds
#endif
#ifdef ENABLE_BVH
                      , bvhNodes
#endif
                      , intersect.hitGeomId
                      , rng);
                    pathSegment.remainingBounces--;
                }
            } else {
                if (environmentMap != NULL) {
                    pathSegment.color *= environmentMap[0].getEnvironmentColor(pathSegment.ray.direction);
                } else {
                    pathSegment.color = glm::vec3(0.0f);
                }
                pathSegment.remainingBounces = 0;
            }
            

        }
    }
}


// Add the current iteration's output to the overall image
__global__
void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < nPaths) {
        PathSegment tmp = iterationPaths[index];
        image[index] += (tmp.color + glm::vec3(0.1f, 0.1f, 0.1f));
    }
}


/**
* Wrapper for the __global__ call that sets up the kernel calls and does a ton
* of memory management
*/
void pathTrace(uchar4* pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelNum = cam.resolution.x * cam.resolution.y;

    //2d block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blockPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y
        );

    //1d block for pathtracing
    const int blockSize1d = 128;

    const int subPixelNum = 1;
    const float subPixelInc = 1.0f / static_cast<float>(subPixelNum);

    
        
    generateRayFromCamera << <blockPerGrid2d, blockSize2d >> >(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate ray from camera");

    int depth = 0;
    PathSegment *dev_path_end = dev_paths + pixelNum;
    int num_paths = pixelNum;


    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    int iterationComplete = 1;
    
    while (iterationComplete < traceDepth) {
        cudaMemset(dev_intersection, 0, sizeof(ShadeableIntersection)* pixelNum);

        dim3 numBlockPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersection << <numBlockPathSegmentTracing, blockSize1d >> >(
            num_paths,
            dev_paths,
            dev_geometry,
            hst_scene->geometrys.size(),
            dev_intersection,
            dev_tris,
            dev_curves
#ifdef ENABLE_MESHWORLDBOUND
            ,dev_worldBounds
#endif
#ifdef ENABLE_BVH
            ,dev_bvhNodes
#endif
            );
        checkCUDAError("compute intersection");

        cudaDeviceSynchronize();
        ++depth;


        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.
        shadeMaterial << <numBlockPathSegmentTracing, blockSize1d >> >(
            depth,
            iter,
            num_paths,
            dev_intersection,
            dev_paths,
            dev_material,
            dev_geometry,
            dev_tris,
            dev_environmentMap,
            dev_textureMap
#ifdef ENABLE_MESHWORLDBOUND
           ,dev_worldBounds
#endif
#ifdef ENABLE_BVH
           ,dev_bvhNodes
#endif
            );

        iterationComplete++;// TODO: should be based off stream compaction results.
        
    }

    dim3 numBlockPixels = (pixelNum + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlockPixels, blockSize1d >> >(
        num_paths,
        dev_image,
        dev_paths);

    checkCUDAError("gather image");
    
    sendImageToPBO << <blockPerGrid2d, blockSize2d >> > (
        pbo,
        cam.resolution,
        iter * subPixelNum,
        dev_image);

    checkCUDAError("image to pbo");

    //save image to renderstate
    cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelNum * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("path trace");


}



