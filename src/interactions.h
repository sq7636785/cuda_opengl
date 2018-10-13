#pragma once

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include "intersections.h"

//#define PBR



__host__ __device__
glm::vec3 SampleSphereUniform(float random_x, float random_y) {
    float z = 1.0f - 2 * random_x;

    float x = cos(2.0f * PI * random_y) * sqrt(1.0f - z * z);
    float y = sin(2.0f * PI * random_y) * sqrt(1.0f - z * z);

    return glm::vec3(x, y, z);
}
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


__host__ __device__ inline bool Refract(const glm::vec3 &wi, const glm::vec3 &n, float eta,
    glm::vec3 *wt) {
    // Compute cos theta using Snell's law
    float cosThetaI = glm::dot(n, wi);
    //float sin2ThetaI = std::max(float(0), float(1 - cosThetaI * cosThetaI));
    float sin2ThetaI = fmaxf(0.f, (1.0f - cosThetaI * cosThetaI));
    float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1) return false;
    float cosThetaT = std::sqrt(1 - sin2ThetaT);
    *wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
    return true;
}

__host__ __device__
glm::vec3 fractRay(PathSegment &pathSegment, glm::vec3 normal, float prob, float ratio) {

    glm::vec3 direction;
    float NI = glm::dot(normal, pathSegment.ray.direction);
    
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


__host__ __device__ 
inline glm::vec3 Faceforward(const glm::vec3 &n, const glm::vec3 &v) {
    return (glm::dot(n, v) < 0.f) ? -n : n;
}

__host__ __device__
void isotropicScatterintMedium(
    Geometry mediumGeom,
    int& remainingBounds,
    glm::vec3 &color,
    glm::vec3 &ori,
    glm::vec3 &dir,
    thrust::default_random_engine &rng,
    Triangle* tris
#ifdef ENABLE_MESHWORLDBOUND
  , Bounds3f* worldBounds
#endif
#ifdef ENABLE_BVH
  , LinearBVHNode* bvhNodes
#endif
) {
    float tFar;

    Ray rayInMedium;
    rayInMedium.origin = ori,
    rayInMedium.direction = dir;

    thrust::uniform_real_distribution<float> u01(0, 1);

    glm::vec3 tmp_normal;
    glm::vec3 tmpIntersectPoint;
    bool tmp_outside;

    // tweak this parameter
    glm::vec3 absorptionColor(0.4f, 0.4f, 0.4f); // >1, start feeling like emitting

    // tweak this parameter
    float absorptionAtDistance = 5.5f; // has something to do with density. 
    //Larger -> lighter -> less enegy absorbed
    //Small  ->  darker -> more enegy absorbed

    //// This method for calculating the absorption coefficient is borrowed from Burley's 2015 Siggraph Course Notes "Extending the Disney BRDF to a BSDF with Integrated Subsurface Scattering"
    //// It's much more intutive to specify a color and a distance, then back-calculate the coefficient
    glm::vec3 absorptionCoefficient = (-1.0f / absorptionAtDistance) *
        glm::vec3(log10f(absorptionColor.x),
        log10f(absorptionColor.y),
        log10f(absorptionColor.z));

    float scatteringDistance = 0.25f; // tweak this parameter
    float scatteringCoefficient = (1.0f / scatteringDistance);
    bool isBounceInsideMediumFinish = false;

    //To set up, set tranmission totally 1 -> no absorb
    glm::vec3 Transmission(1.0f, 1.0f, 1.0f);

    //int bounceTime = 0;
    //int maxBounceTime = 10;

    // --------------- scattering start here -------------------
    while (true) {
        ShadeableIntersection temp_isect;
        temp_isect.t = FLT_MAX;
        int hit_tri_index = -1;

        //if (bounceTime >= maxBounceTime) {

        //	//Move Ray outside medium
        //	IntersectBVH(rayInMedium, &temp_isect, hit_tri_index, nodes, tris);
        //	tFar = temp_isect.t;
        //	rayInMedium.origin += (tFar + 0.0005f) * rayInMedium.direction;
        //	break; 
        //}

        // tFar is the largest distance from the origin of rayInMedium now to the exit of medium

        //Sphere intersect test
        if (mediumGeom.type == SPHERE) {
            tFar = sphereIntersectionTest(mediumGeom, rayInMedium, tmpIntersectPoint, tmp_normal, tmp_outside);
        }

        //cube intersect test
        if (mediumGeom.type == CUBE) {
            tFar = boxIntersectionTest(mediumGeom, rayInMedium, tmpIntersectPoint, tmp_normal, tmp_outside);
        }

        //mesh intersect test
        if (mediumGeom.type == MESH) {
            if (intersectBVH(rayInMedium, &temp_isect, hit_tri_index, bvhNodes, tris)) {
                if (hit_tri_index >= mediumGeom.startIndex
                    && hit_tri_index < mediumGeom.endIndex) {
                    tFar = temp_isect.t;
                } else {
                    tFar = -1.0f;
                }
            }
        }

        if (tFar < 0.0f) {
            // it means that rayInMedium has been outside the medium(geom)
            break; //Exit
        }

        //float weight = 1.0f;
        //float pdf = 1.0f;

        //1.sample distance
        float random_float = u01(rng);
        float distance = -logf(random_float) / scatteringCoefficient;
        //float distance = -logf(random_float) / density;

        // If we sample a distance farther than the next intersecting surface, clamp to the surface distance
        if (distance >= tFar) {
            //pdf = 1.0f;
            distance = tFar;
            isBounceInsideMediumFinish = true;
        } else {
            //pdf = std::exp(-scatteringCoefficient * distance);
        }

        //2.get transmission of sampled distance
        Transmission *= glm::vec3(expf(-absorptionCoefficient.x * distance),
            expf(-absorptionCoefficient.y * distance),
            expf(-absorptionCoefficient.z * distance));


        //3.move ray along distance sampled
        rayInMedium.origin += rayInMedium.direction * distance;

        //4.uniformly sample a new ray direction

        //rayInMedium.direction = calculateRandomDirectionInHemisphere(rayInMedium.direction, rng);
        //rayInMedium.direction = glm::normalize(rayInMedium.direction);

        float random_x = u01(rng);
        float random_y = u01(rng);
        rayInMedium.direction = glm::normalize(SampleSphereUniform(random_x, random_y));

        if (isBounceInsideMediumFinish) {
            // This means ray has hit the exit of medium(geom)
            rayInMedium.origin += 0.001f * rayInMedium.direction;
            break; // Exit
        }

        if (Transmission.x < 0.05f &&
            Transmission.y < 0.05f &&
            Transmission.z < 0.05f) {
            //if Transmission is very small(totally absorbed)
            //no need to trace this ray any more
            remainingBounds = 0;
            break;
        }

        //bounceTime++;
    }

    color *= (Transmission);

    ori = rayInMedium.origin;
    dir = rayInMedium.direction;
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
    Material &m,
    Geometry* geoms,
    Triangle* tris
#ifdef ENABLE_MESHWORLDBOUND
   ,Bounds3f* worldBounds
#endif
#ifdef ENABLE_BVH
   ,LinearBVHNode* bvhNodes
#endif
   ,int hitGeomsID
   ,thrust::default_random_engine &rng
    ) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    //diffuse
    thrust::uniform_real_distribution<float> u01(0, 1);
    float prob = u01(rng);
    glm::vec3 wi;
    
    bool isGlass = (m.hasReflective > 0.0f && m.hasRefractive > 0.0f);

    //subsurface scattering
    if (m.isBssdf) {
        
        glm::vec3 rOrigin = intersect + pathSegment.ray.direction * 0.0005f + 0.0002f * Faceforward(normal, pathSegment.ray.direction);
        glm::vec3 newDirection = pathSegment.ray.direction; 
        isotropicScatterintMedium(geoms[hitGeomsID], pathSegment.remainingBounces, pathSegment.color, rOrigin, newDirection, rng, tris
#ifdef ENABLE_MESHWORLDBOUND
            , worldBounds
#endif
#ifdef ENABLE_BVH
            , bvhNodes
#endif
            );
        pathSegment.ray.origin = rOrigin;
        pathSegment.ray.direction = newDirection;
    } 
    
    //glass, reflect and refract
    else if (isGlass) {
        float prob = u01(rng);
        wi = fractRay(pathSegment, normal, prob, m.indexOfRefraction);
        pathSegment.color *= m.specular.color;
        pathSegment.ray.origin = intersect + wi * 0.001f;
        pathSegment.ray.direction = wi;
        pathSegment.color *= m.color;
    } 

    //pure refract
    else if (m.hasRefractive > 0.0f) {
       
        bool enter = glm::dot(normal, pathSegment.ray.direction) < 0.0f;
        float ratio = m.indexOfRefraction;
        if (enter) {
            ratio = 1.f / ratio;
        }
        wi = glm::refract(pathSegment.ray.direction, normal, ratio);
        pathSegment.color *= m.specular.color;
        pathSegment.ray.origin = intersect + wi * 0.001f;
        pathSegment.ray.direction = wi;
        pathSegment.color *= m.color;
    } 
    
    //pure reflect
    else if (m.hasReflective > 0.0f) {
        wi = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
        pathSegment.color *= m.specular.color;
        pathSegment.ray.origin = intersect + wi * 0.001f;
        pathSegment.ray.direction = wi;
        pathSegment.color *= m.color;
    }
    else {
        if (m.specular.exponent > 0.0f) {
            //glossy
            wi = ggxImportanceSample(normal, rng, m.specular.exponent);
        }
        else {//lambert
            wi = calculateRandomDirectionInHemisphere(normal, rng);
        }
        pathSegment.ray.origin = intersect + wi * 0.001f;
        pathSegment.ray.direction = wi;
        pathSegment.color *= m.color;
    }

    /*
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
    }*/
    //pathSegment.remainingBounces--;
    
    
}
