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
    MESH,
    CURVE
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
    int				curveId;
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
    int             textureId;
    bool            isBssdf;
};


struct Camera {
    glm::ivec2  resolution;
    glm::vec3   position;
    glm::vec3   lookAt;
    glm::vec3   view;
    glm::vec3   up;
    glm::vec3   right;
    glm::vec2   fov;
    glm::vec2   pixelLength;

    float       lenRadius;
    float       focalDistance;

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
    glm::vec2   uv;
    glm::vec3   intersectPoint;
    glm::mat3   tangentToWorld;
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
        Bounds3f worldBounds();
    __host__ __device__
        float surfaceArea() {
        return glm::length(glm::cross(vertices[0] - vertices[1], vertices[2] - vertices[1])) * 0.5f;
    }

    __host__ __device__ 
        bool Intersect(const Ray& r, ShadeableIntersection* isect) const;
};



struct CurveCommon {
    CurveCommon(const glm::vec3 c[4], float width0, float width1, const glm::vec3 *n) {
        controlPoint[0] = c[0];
        controlPoint[1] = c[1];
        controlPoint[2] = c[2];
        controlPoint[3] = c[3];
        startWidth = width0;
        endWidth = width1;
        if (n) {
            normal[0] = n[0];
            normal[1] = n[1];
        }
    }
    glm::vec3	controlPoint[4];
    glm::vec3	normal[2];
    float		startWidth;
    float		endWidth;
};


struct Curve {
    Curve() {
        common = nullptr;
    };
    Curve(float min, float max, CurveCommon* c) {
        uMin = min;
        uMax = max;
        common = c;
    };
    
        Bounds3f objBound() const;
    __host__ __device__
        bool Intersect(const Ray& ray, float *tHit, ShadeableIntersection *isect, glm::mat4 worldToObj, glm::mat4 objToWorld);
    __host__ __device__
        bool recursiveIntersect(const Ray &ray, float *tHit,
            ShadeableIntersection *isect, const glm::vec3 cp[4],
            const glm::mat4 &rayToObject, glm::mat4 &objToWorld, float u0, float u1,
            int depth) const;

    float							uMin;
    float							uMax;
    CurveCommon*					common;
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
        glm::vec3 getColor(glm::vec2& uv);
    __host__ __device__
        glm::vec3 getNormal(glm::vec2& uv);

    __host__ __device__
        glm::vec3 getEnvironmentColor(glm::vec3& dir);
};


#endif // !DATA_STRUCTURE
