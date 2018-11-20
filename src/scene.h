#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

#include "glm/glm.hpp"
#include "utilities.h"
#include "data_structure.h"

#include "bvh.h"

#define ENABLE_MESHWORLDBOUND

#define ENABLE_BVH

class Scene {
  private:
      std::ifstream fp_in;

      int loadMaterial(std::string fileName);
      int loadGeometry(std::string fileName);
      int loadCamera();
      int loadEnvironment();
      int loadObj(std::string objPath, Geometry &newGeom);
      int loadCurve(std::string curvePath, Geometry &newGeom);
      std::vector<Curve> makeCurve(const glm::vec3* c, float w0, float w1, int sd = 8);

  public:
      Scene(const std::string &fileName);
      ~Scene();

      std::vector<Geometry> geometrys;
      std::vector<Material> materials;
      std::vector<Triangle> triangles;
      std::vector<Bounds3f> worldBounds;
      std::vector<Texture>  environmentMap;
      std::vector<Texture>  textureMap;
      std::vector<Curve>	curves;
      RenderState           state;

      

#ifdef ENABLE_BVH
      LinearBVHNode*         bvhNodes;
      int                    bvhTotalNodes;
#endif
};

#endif