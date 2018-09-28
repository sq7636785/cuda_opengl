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