#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "glslUtility.h"
#include "data_structure.h"
#include "image.h"
#include "path_tracer.h"
#include "utilities.h"
#include "scene.h"

extern Scene* scene;
extern int    iteration;
extern int    width;
extern int    height;

void runCuda();
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);


#endif