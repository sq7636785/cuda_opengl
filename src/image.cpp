#include <iostream>
#include <string>
#include "image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

image::image(int x, int y)
:xSize(x), ySize(y),
pixels(new glm::vec3[x * y]) {
}

image::~image() {
    delete pixels;
}


void image::setPixel(int x, int y, glm::vec3 &pixel) {
    assert(x >= 0 && y >= 0 && x < xSize && y < ySize);
    pixels[x + xSize * y] = pixel;
}


void image::savePNG(const std::string &baseFileName) {
    unsigned char *bytes = new unsigned char[3 * xSize * ySize];
    for (int y = 0; y < ySize; ++y) {
        for (int x = 0; x < xSize; ++x) {
            int i = y * xSize + x;
            glm::vec3 pix = glm::clamp(pixels[i], glm::vec3(), glm::vec3(1)) * 255.0f;
            bytes[i * 3 + 0] = static_cast<unsigned char>(pix.x);
            bytes[i * 3 + 1] = static_cast<unsigned char>(pix.y);
            bytes[i * 3 + 2] = static_cast<unsigned char>(pix.z);
        }
    }
    std::string fileName = baseFileName + ".png";
    stbi_write_png(fileName.c_str(), xSize, ySize, 3, bytes, xSize * 3);
    std::cout << "save png files " << fileName << std::endl;

    delete[] bytes;
}

void image::saveHDR(const std::string &baseFileName) {
    std::string fileName = baseFileName + ".hdr";
    stbi_write_hdr(fileName.c_str(), xSize, ySize, 3, reinterpret_cast<float*>(pixels));
    std::cout << "save hdr files " << fileName << std::endl;
}

