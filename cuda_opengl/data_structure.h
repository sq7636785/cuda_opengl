#ifndef DATA_STRUCTURE
#define DATA_STAUCTURE

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

enum GeomType {
    SPHERE,
    CUBE,
};

struct Ray {
    glm::vec3 position; 
    glm::vec3 diretion;
};

struct Geometry {
    enum GeomType type;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Material {
    glm::vec3 color;
    struct {
        float       exponent;
        glm::vec3   color;
    }specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
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


struct RenderState {
    Camera       camera;
    unsigned int iterations;
    int          traceDepth;
    std::vector<glm::vec3> image;
    std::string  imageName;
};

struct ShadeableIntersection {
    float       t;
    glm::vec3   surfaceNormal;
    int         materialId;
};

#endif // !DATA_STRUCTURE
