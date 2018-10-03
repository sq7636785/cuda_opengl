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
        color.x = glm::clamp(static_cast<int>(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp(static_cast<int>(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp(static_cast<int>(pix.z / iter * 255.0), 0, 255);

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

    checkCUDAError("pathTraceInit");
}


void pathTraceFree() {
    cudaFree(dev_geometry);
    cudaFree(dev_material);
    cudaFree(dev_image);
    cudaFree(dev_intersection);
    cudaFree(dev_paths);

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
        PathSegment &tmp = pathSegments[index];
        tmp.ray.position = cam.position;
        tmp.remainingBounces = traceDepth;
        tmp.pixelIndex = index;
        tmp.color = glm::vec3(1.0, 1.0, 1.0);
        // TODO: implement antialiasing by jittering the ray
        tmp.ray.diretion = glm::normalize(cam.view - (static_cast<float>(x)-cam.resolution.x * 0.5f) * cam.pixelLength.x * cam.right
                                                   - (static_cast<float>(y)-cam.resolution.y * 0.5f) * cam.pixelLength.y * cam.up);
    }
}


// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__
void computeIntersection(
    int depth, 
    int num_paths, 
    PathSegment* pathSegments, 
    Geometry* geoms, 
    int geoms_size, 
    ShadeableIntersection *intersections) {
    
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < num_paths) {
        //final para
        float tMin = 10000.0f;
        glm::vec3 tNormal;
        int geomsId = -1;
        PathSegment &unit = pathSegments[index];
        ShadeableIntersection &si = intersections[index];
        //tmp para
        bool outside = true;
        glm::vec3 intersectPoint;
        glm::vec3 normal;
        float t;

        for (int i = 0; i < geoms_size; ++i) {
            if (geoms[i].type == GeomType::SPHERE) {
                t = sphereIntersectionTest(geoms[i], unit.ray, intersectPoint, normal, outside);
            } else if (geoms[i].type == GeomType::CUBE) {
                t = boxIntersectionTest(geoms[i], unit.ray, intersectPoint, normal, outside);
            }
            //TODO:  more geometry type
            
            if (t < tMin && t > 0) {
                tMin = t;
                tNormal = normal;
                geomsId = i;
            }
        }
        if (tMin > 0 && geomsId != -1) {
            si.materialId = geoms[geomsId].materialID;
            si.surfaceNormal = tNormal;
            si.t = tMin;
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


// Add the current iteration's output to the overall image
__global__
void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < nPaths) {
        PathSegment tmp = iterationPaths[index];
        image[index] += tmp.color;
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
    generateRayFromCamera << <blockPerGrid2d, blockSize2d >> >(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate ray from camera");

    int depth = 0;
    PathSegment *dev_path_end = dev_paths + pixelNum;
    int num_paths = pixelNum;


    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete) {
        cudaMemset(dev_intersection, 0, sizeof(ShadeableIntersection)* pixelNum);

        dim3 numBlockPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersection << <numBlockPathSegmentTracing, blockSize1d >> >(
            depth,
            num_paths,
            dev_paths,
            dev_geometry,
            hst_scene->geometrys.size(),
            dev_intersection
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
        shadeFakeMaterial << <numBlockPathSegmentTracing, blockSize1d >> >(
            iter,
            num_paths,
            dev_intersection,
            dev_paths,
            dev_material);

        iterationComplete = true;// TODO: should be based off stream compaction results.
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
        iter,
        dev_image);

    checkCUDAError("image to pbo");

    //save image to renderstate
    cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelNum * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("path trace");


}



