#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

#include "glm/glm.hpp"
#include "utilities.h"
#include "data_structure.h"

class Scene {
  private:
      std::ifstream fp_in;

      int loadMaterial(const std::string &fileName);
      int loadGeometry(const std::string &fileName);
      int loadCamera();

  public:
      Scene(const std::string &fileName);
      ~Scene();

      std::vector<Geometry> geometrys;
      std::vector<Material> materials;
      RenderState           state;
};

#endif