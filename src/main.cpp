#include <iostream>

#include "glm/glm.hpp"
#include "scene.h"

using namespace std;

int main() {
    const char* sceneFile = "cornell.txt";
    Scene *scene = new Scene(sceneFile);

    return 0;
}