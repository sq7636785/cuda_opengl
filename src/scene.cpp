#include <iostream>
#include <cstring>

#include "glm/gtc/matrix_inverse.hpp"
#include "glm/gtx/string_cast.hpp"
#include "scene.h"

Scene::Scene(const std::string &fileName) {
    std::cout << "reading files " << fileName << "..." << std::endl;
    fp_in.open(fileName.c_str());
    if (!fp_in.is_open()) {
        std::cout << "Error reading from file - aborting!" << std::endl;
        throw;
    }

    while (fp_in.good()) {
        std::string line;
        utilityCore::safeGetline(fp_in, line);
        
        if (!line.empty()) {
            std::vector<std::string> tokens = utilityCore::tokenizeString(line);

            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
            }
            if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeometry(tokens[1]);
            }
            if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
            }
        }
    }
}


int Scene::loadGeometry(const std::string &fileName) {
    int id = atoi(fileName.c_str());
    if (id != geometrys.size()) {
        std::cout << "ERROR: OBJECT ID does not match expected number of geoms" << std::endl;
        return -1;
    } else {
        std::cout << "Load Geometry " << id << std::endl;
        Geometry newGeom;
        std::string line;

        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                std::cout << "creating new sphere" << std::endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                std::cout << "creating new cube" << std::endl;
                newGeom.type = CUBE;
            }
        }


        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            std::vector<std::string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialID = atoi(tokens[1].c_str());
            std::cout << "connect object " << id << " to material " << newGeom.materialID << std::endl;
        }

        //load transform
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            std::vector<std::string> tokens = utilityCore::tokenizeString(line);

            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geometrys.push_back(newGeom);
        return 1;
    }
}



int Scene::loadMaterial(const std::string &fileName) {
    int id = atoi(fileName.c_str());
    if (id != materials.size()) {
        std::cout << "ERROR: MATERIAL ID does not match expected number of materials" << std::endl;
        return -1;
    } else {
        std::cout << "Load Material " << id << "..." << std::endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 7; ++i) {
            std::string line;
            utilityCore::safeGetline(fp_in, line);
            std::vector<std::string> tokens = utilityCore::tokenizeString(line);

            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.color = color;
            } else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 color(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = color;
            } else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            }
        } 
        materials.push_back(newMaterial);
        return 1;
    }

}


int Scene::loadCamera() {
    RenderState &state = this->state;
    Camera &camera = state.camera;

    float fovy;

    //load static propertity
    for (int i = 0; i < 5; ++i) {
        std::string line;
        utilityCore::safeGetline(fp_in, line);
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);

        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    std::string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);

        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }


    //calcute fov
    
    /*
    my origin calcution
    float tanyh = tan(0.5f * fovy * (PI / 180.0f));
    float focalLength = 0.5f * camera.resolution.y / tanyh;
    float fovx = 2.0f * atan( 0.5f * camera.resolution.x / focalLength ) * 180.0f / PI;
    camera.fov = glm::vec2(fovx, fovy);

    //calcute pixel increase for the ray direction
    float incY = tanyh * focalLength * 2 / camera.resolution.y;
    float incX = incY * camera.resolution.x / camera.resolution.y;
    //here you get incY, if you want to calucute the new ray direction
    // focalPoint + incY + incX - camPos, then you normalize it
    // so, there are many optimal steps. you can cancel focal point and focallength.
    // because you only need normalized vector.
    */

    //tan(fovy/2) / (y / 2) = tan (fovx/2) / (x / 2), so, the divide 2 can be saved.
    float yScale = tan(fovy * PI / 180.0f);
    float xScale = yScale * camera.resolution.x / camera.resolution.y;
    float fovx   = atan(xScale) * 180.0f / PI;
    camera.fov   = glm::vec2(fovx, fovy);

    //we set focallenth to 1, the the focalpoint(0, 0), so the up point is (0, tan(fovy/2) * foallenth)
    //so each pixel increse is tan(fovy/2) * focallength / (y / 2)
    // ps: because you assump the focallenth is 1. so you should strict normalize the cam.view. 
    // because you need calculate the new direction by glm::normalize(cam.view + incX * camera.right + incY * camera.up)
    float xInc = xScale * 2.0f / camera.resolution.x;
    float yInc = yScale * 2.0f / camera.resolution.y;
    camera.pixelLength = glm::vec2(xInc, yInc);
    
    //在视点坐标系中， n(z), u(y), v(x)， n是视线方向的负方向， 因为这样才和opengl的定义一致， 以及之后的投影操作符合
    //但可以利用视线方向来计算v和u，
    //但在最后返回视点坐标系矩阵时候， 应该将视线方向取负作为视线坐标系中的n轴
    //在平移上， 可以看下你总结的坐标变换， 就是求视点位置在视点坐标系下各个轴的投影。
    
    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.up = glm::normalize(glm::cross(camera.right, camera.view));

    //image vector
    unsigned int pixelNum = static_cast<unsigned int>(camera.resolution.x * camera.resolution.y);
    state.image.assign(pixelNum, glm::vec3());

    std::cout << "load camera" << std::endl;

    return 1;
}