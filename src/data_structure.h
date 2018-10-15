#ifndef DATA_STAUCTURE_H
#define DATA_STAUCTURE_H

#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "glm/gtx/intersect.hpp"
#include "bounds.h"


#include <stb_image.h>


#define COLORDIVIDOR 0.003921568627f
#define InvPi 0.31830988618379067154f
#define Inv2Pi 0.15915494309189533577f

#define Pi 3.14159265358979323846f


enum GeomType {
    SPHERE,
    CUBE,
    MESH
};


struct Geometry {
    enum GeomType   type;
    int             materialID;
    glm::vec3       translation;
    glm::vec3       rotation;
    glm::vec3       scale;
    glm::mat4       transform;
    glm::mat4       inverseTransform;
    glm::mat4       invTranspose;

    int             startIndex;
    int             endIndex;
    int             worldBoundIdx;
};

struct Material {
    struct {
        float       exponent;
        glm::vec3   color;
    }specular;

    glm::vec3       color;
    float           hasReflective;
    float           hasRefractive;
    float           indexOfRefraction;
    float           emittance;
    bool            isBssdf;
};


struct Camera {
    glm::ivec2 resolution;
    glm::vec3  position;
    glm::vec3  lookAt;
    glm::vec3  view;
    glm::vec3  up;
    glm::vec3  right;
    glm::vec2  fov;
    glm::vec2  pixelLength;
};


//record the render state every iteration
struct RenderState {
    Camera                  camera;
    unsigned int            iterations;
    int                     traceDepth;
    std::vector<glm::vec3>  image;
    std::string             imageName;
};

//thread part
struct PathSegment {
    Ray         ray;
    glm::vec3   color;
    int         pixelIndex;
    int         remainingBounces;
};

struct ShadeableIntersection {
    float       t;
    glm::vec3   surfaceNormal;
    int         materialId;
    int         hitGeomId;
    glm::vec3   intersectPoint;
    glm::vec3   tangentToWorld;
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    Vertex(glm::vec3 v, glm::vec3 n) : position(v), normal(n) {}
};


struct Triangle {
    int index;
    glm::vec3 vertices[3];
    glm::vec3 normals[3];
    glm::vec2 uvs[3];


    __host__ __device__
        Bounds3f worldBounds() {
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
        float surfaceArea() {
        return glm::length(glm::cross(vertices[0] - vertices[1], vertices[2] - vertices[1])) * 0.5f;
    }

    __host__ __device__ bool Intersect(const Ray& r, ShadeableIntersection* isect) const {

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
};


struct Texture {
    int             width;
    int             height;
    int             nComp;
    unsigned char*  hostData;
    unsigned char*  devData;


    void loadFromFile(const char* fileName) {
        hostData = stbi_load(fileName, &width, &height, &nComp, 0);
        if (hostData == nullptr) {
            std::cout << "ERROR : texture load fail!" << std::endl;
        }
        devData = nullptr;
    }

    void free() {
        stbi_image_free(hostData);
    }

    __host__ __device__
        glm::vec3 getColor(glm::vec2& uv) {
        float w = static_cast<float>(width);
        float h = static_cast<float>(height);
        int x = glm::min(w * uv.x, w - 1.0f);
        int y = glm::min(h * (1.0f - uv.y), h - 1.0f);

        int texelIdx = y * width + x;
        //gpu only
        glm::vec3 color = COLORDIVIDOR * glm::vec3(devData[texelIdx * nComp], devData[texelIdx * nComp + 1], devData[texelIdx * nComp + 2]);
        return color;
    }

    __host__ __device__
        glm::vec3 getNormal(glm::vec2& uv) {
        float w = static_cast<float>(width);
        float h = static_cast<float>(height);
        int x = glm::min(w * uv.x, w - 1.0f);
        int y = glm::min(h * ( 1.0f - uv.y), h - 1.0f);

        int texelIdx = y * width + x;
        glm::vec3 normal = glm::vec3(devData[texelIdx * nComp], devData[texelIdx * nComp + 1], devData[texelIdx * nComp + 2]);
        normal = 2.0f * COLORDIVIDOR * normal;
        normal = glm::vec3(normal.x - 1.0f, normal.y - 1.0f, normal.z - 1.0f);
        return normal;
    }

    __host__ __device__
        glm::vec3 getEnvironmentColor(glm::vec3& dir) {
        dir = glm::normalize(dir);
        float phi = std::atan2(dir.z, dir.x);
        phi = (phi < 0.0f) ? (phi + 2.0f * Pi) : phi;
        float theta = glm::acos(dir.y);

        glm::vec2 uv = glm::vec2(phi * Inv2Pi, 1.0f - theta * InvPi);
        return getColor(uv);
    }
};


#endif // !DATA_STRUCTURE
