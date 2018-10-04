#pragma once

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include "intersections.h"

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
glm::vec3 fractRay(PathSegment &pathSegment, glm::vec3 normal) {
    glm::vec3 wi;// = glm::normalize(glm::refract(pathSegment.ray.diretion, normal, 0.7f));

    bool into = glm::dot(pathSegment.ray.diretion, normal) < 0;
    float nc = 1, nt = 1.5;
    float nnt = into ? nc / nt : nt / nc;
    float ddn = glm::abs(glm::dot(pathSegment.ray.diretion, normal));
    float cos2t;

    //total internal reflection
    if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) < 0) {
        wi = glm::normalize(glm::reflect(pathSegment.ray.diretion, normal));
    } else {
        wi = glm::normalize((pathSegment.ray.diretion * nnt - normal * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))));
    }
    return wi;
}

__host__ __device__
void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng,
    bool firstHit) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    //diffuse
    thrust::uniform_real_distribution<float> u01(0, 1);
    float prob = u01(rng);
    glm::vec3 direction;

    if (m.emittance > 0) {
        pathSegment.color *= (m.emittance * m.color);
        pathSegment.remainingBounces = 0;
    } else {
        
        if (m.hasReflective) {
            
                glm::vec3 wi = glm::normalize(glm::reflect(pathSegment.ray.diretion, normal));
            
            pathSegment.color *= m.specular.color;
            pathSegment.ray.position = intersect + wi * 0.01f;
            pathSegment.ray.diretion = wi;
            
            pathSegment.remainingBounces--;

        } else if (m.hasRefractive) {
            
            glm::vec3 wi = fractRay(pathSegment, normal);
            //glm::vec3 wi = glm::normalize(glm::refract(pathSegment.ray.diretion, normal, 1.0f / 1.5f));
            pathSegment.ray.position = intersect + wi * 0.01f;
            pathSegment.ray.diretion = wi;
            pathSegment.color *= m.specular.color;
            pathSegment.remainingBounces--;
        } else {
            //lambert
            glm::vec3 wi = calculateRandomDirectionInHemisphere(normal, rng);
            float cosTheta = glm::abs(glm::dot(normal, wi));
            pathSegment.ray.position = intersect + wi *0.01f;
            pathSegment.ray.diretion = wi;
            pathSegment.color *= m.color;
            if (firstHit) { pathSegment.color *= cosTheta; }
            pathSegment.remainingBounces--;
        }
    }
    
}
