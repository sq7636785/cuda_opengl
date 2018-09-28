#ifndef IMAGE_H
#define IMAGE_H

#include "glm/glm.hpp"

class image {
  private:
      int        xSize;
      int        ySize;
      glm::vec3  *pixels;

  public:
      image(int x, int y);
      ~image();

      void setPixel(int x, int y, glm::vec3 &pixel);
      void savePNG(const std::string &baseFileName);
      void saveHDR(const std::string &baseFileName);
};

#endif