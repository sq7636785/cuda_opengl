#ifndef DATA_STAUCTURE_H
#define DATA_STAUCTURE_H

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "glm/gtx/intersect.hpp"
#include "bounds.h"

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

    __host__ __device__
        float intersect(const Ray &r, glm::vec3 &intersectPoint, glm::vec3 &normal, glm::mat4 &transform, glm::mat4 &invTransform, bool &outside) const {
        glm::vec3 baryPosition(0.0f);
        Ray ray;
        ray.origin = glm::vec3(invTransform * glm::vec4(r.origin, 1.0f));
        ray.direction = glm::normalize(glm::vec3(invTransform * glm::vec4(r.direction, 0.0f)));

        float t = 0.0f;
        if (glm::intersectRayTriangle(ray.origin, ray.direction, vertices[0], vertices[1], vertices[2], baryPosition)) {
            
            normal = normals[0] * (1.0f - baryPosition.x - baryPosition.y) +
                normals[1] * baryPosition.x + normals[2] * baryPosition.y;
            normal = glm::normalize(normal);
            intersectPoint = r.origin + baryPosition.z * glm::normalize(r.direction);
            intersectPoint = glm::vec3(transform * glm::vec4(intersectPoint, 1.0f));
            normal = glm::vec3(transform * glm::vec4(normal, 0.0f));

            outside = false;
            t = baryPosition.z;
        } else {
            outside = true;
            t = -1.0f;
        }
        return t;
            
    }
};

#endif // !DATA_STRUCTURE
