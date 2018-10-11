#pragma once

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include "intersections.h"

//#define PBR

// CHECKITOUT
/**
* Computes a cosine-weighted random direction in a hemisphere.
* Used for diffuse lighting.
*/
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }
//     if (abs(normal.z) > 0.999f) {
//         directionNotNormal = glm::vec3(1.0f, 0.0f, 0.0f);
//     } else {
//         directionNotNormal = glm::vec3(0.0f, 1.0f, 0.0f);
//     }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}



__host__ __device__
glm::vec3 ggxImportanceSample(glm::vec3 normal, thrust::default_random_engine &rng, float exponent) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float u1 = u01(rng);
    float u2 = u01(rng);
    float phi = u1 * TWO_PI;
    float cosTheta = sqrt((1.0f - u2) / ((exponent * exponent - 1.0f) * u2 + 1.0f));
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }
    glm::vec3 worldX = glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 worldY = glm::normalize(glm::cross(normal, worldX));
    return sinTheta * cos(phi) * worldX + sinTheta * sin(phi) * worldY + cosTheta * normal;
}

__host__ __device__
glm::vec3 fractRay(PathSegment &pathSegment, glm::vec3 normal, float prob) {

    glm::vec3 direction;
    float NI = glm::dot(normal, pathSegment.ray.direction);
    float ratio = 1.5f;
    if (NI < 0) {
        ratio = 1.f / ratio;
    }
    float r0 = (1.f - ratio) / (1.f + ratio);
    r0 *= r0;
    float x = 1.f + NI;
    float r = r0 + (1.f - r0) * x * x * x * x * x;

    if (prob < r) {
        direction = glm::reflect(pathSegment.ray.direction, normal);
    } else {
        direction = glm::refract(pathSegment.ray.direction, normal, ratio);
    }
    return direction;
}


__host__ __device__
float distributionGGX(glm::vec3 N, glm::vec3 H, float roughness) {
    float a = roughness * roughness;
    float NdotH = glm::max(glm::dot(N, H), 0.0f);
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (a - 1.0) + 1.0);

    return a / (PI * denom * denom);
}

__host__ __device__
float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0f;
    float k = r * r / 8.0f;
    return NdotV / (NdotV * (1.0f - k) + k);
}

__host__ __device__
float geometrySmith(glm::vec3 N, glm::vec3 V, glm::vec3 L, float roughness) {
    float NdotV = glm::max(glm::dot(N, V), 0.0f);
    float NdotL = glm::max(glm::dot(N, L), 0.0f);

    return geometrySchlickGGX(NdotV, roughness) * geometrySchlickGGX(NdotL, roughness);
}

__host__ __device__
glm::vec3 fresnelSchlickRoughness(float cosTheta, glm::vec3 F0, float roughness) {
    return F0 + (glm::max(glm::vec3(1.0 - roughness), F0) - F0) * pow(1.0f - cosTheta, 5.0f);
}

/**
* Scatter a ray with some probabilities according to the material properties.
* For example, a diffuse surface scatters in a cosine-weighted hemisphere.
* A perfect specular surface scatters in the reflected ray direction.
* In order to apply multiple effects to one surface, probabilistically choose
* between them.
*
* The visual effect you want is to straight-up add the diffuse and specular
* components. You can do this in a few ways. This logic also applies to
* combining other types of materias (such as refractive).
*
* - Always take an even (50/50) split between a each effect (a diffuse bounce
*   and a specular bounce), but divide the resulting color of either branch
*   by its probability (0.5), to counteract the chance (0.5) of the branch
*   being taken.
*   - This way is inefficient, but serves as a good starting point - it
*     converges slowly, especially for pure-diffuse or pure-specular.
* - Pick the split based on the intensity of each material color, and divide
*   branch result by that branch's probability (whatever probability you use).
*
* This method applies its changes to the Ray parameter `ray` in place.
* It also modifies the color `color` of the ray in place.
*
* You may need to change the parameter list for your purposes!
*/



__host__ __device__
void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    //diffuse
    thrust::uniform_real_distribution<float> u01(0, 1);
    float prob = u01(rng);
    glm::vec3 wi;

    if (m.hasReflective > 0) {
        if (m.specular.exponent == 0.0f) {
            wi = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
        } else {
            wi = glm::normalize(ggxImportanceSample(normal, rng, m.specular.exponent));
        }

        pathSegment.color *= m.specular.color;
    } else if (m.hasRefractive > 0) {
        float prob = u01(rng);
        wi = fractRay(pathSegment, normal, prob);
        pathSegment.color *= m.specular.color;
    } else {
        //lambert
        wi = calculateRandomDirectionInHemisphere(normal, rng);
    }
    if (m.indexOfRefraction == 1.0f) {

        //pbr
        float metallic = 0.2f;
        float F0 = 0.3f;
        float roughness = 0.8f;
        glm::vec3 V = glm::dot(normal, pathSegment.ray.direction) < 0.0f ? -pathSegment.ray.direction : pathSegment.ray.direction;
        glm::vec3 H = glm::normalize(wi + V);
        glm::vec3 F = fresnelSchlickRoughness(glm::max(glm::dot(V, normal), 0.0f), glm::vec3(F0, F0, F0), roughness);
        float NDF = distributionGGX(normal, H, roughness);
        float G = geometrySmith(normal, V, wi, roughness);

        glm::vec3 Ks = F;
        glm::vec3 Kd = (glm::vec3(1.0f, 1.0f, 1.0f) - Ks) * (1 - metallic);

        glm::vec3 specular = (NDF * G * F) / (4 * glm::max(glm::dot(normal, V), 0.0f) * glm::max(glm::dot(normal, wi), 0.0f) + 0.001f);
        glm::vec3 diffuse = Kd / PI;
        float cosTheta = glm::max(glm::dot(normal, wi), 0.0f);
        pathSegment.color *= (diffuse + specular);
    }
    pathSegment.remainingBounces--;
    pathSegment.ray.origin = intersect + wi * 0.001f;
    pathSegment.ray.direction = wi;
    pathSegment.color *= m.color;
}
