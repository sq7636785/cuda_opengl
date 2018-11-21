#include <iostream>

#include "main.h"
#include "preview.h"

using namespace std;

static std::string startTimeString;

//camera relative
static bool         leftMousePressed   = false;
static bool         rightMousePressed  = false;
static bool         middleMousePressed = false;
static double       xPos;
static double       yPos;

static bool         cameraChanged      = true;
static float        dtheta;
static float        dphi;
static glm::vec3    camMov;

float               zoom;
float               theta;
float               phi;
glm::vec3           cameraPosition;
glm::vec3           ogLookAt;      //recenting camera

Scene*              scene;
RenderState*        renderState;
int                 iteration;

int                 width;
int                 height;


int main() {
    //const char* sceneFile = "cornellBox_threeSphere.txt";
    const char* sceneFile = "../scenes/cornell3.txt";
    scene = new Scene(sceneFile);

    iteration = 0;
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    ogLookAt = cam.lookAt;
    cameraPosition = cam.position;
    zoom = glm::length(cameraPosition - ogLookAt);

    /************************************************************************/
    /* problem                                                              */
    /* 这里也应该再调一下，可以按照你的想法改一下
        源码的phi是和z轴的夹角， theta是和y轴的夹角
        这里的theta phi 和下面的交互全是一套的。
        需要仔细理解以下， 或者全按自己的想法来一遍
    */
    /************************************************************************/
    
    //已更正：
    //phi是与正Z轴的夹角。
    //这样根据phi和theta计算出的方向向量就是视线方向
    //乘上focalLength就是lookat到摄像机位置的距离向量

    
    glm::vec3 v = cam.view;
    phi = acos(glm::dot(v, glm::vec3(0.0f, 0.0f, 1.0f)));
    theta = acos(glm::dot(v, glm::vec3(0.0f, 1.0f, 0.0f)));
    
    init();
    mainLoop();


    return 0;
}


void saveImage() {
    float samples = static_cast<float>(iteration);
    image img(width, height);

    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            int index = y * width + x;
            glm::vec3 pixel = renderState->image[index];
            img.setPixel(width - 1 - x, y, pixel / samples);
        }
    }

    std::string fileName = renderState->imageName;
    std::ostringstream ss;
    ss << fileName << "." << startTimeString << "." << samples << "samples";
    fileName = ss.str();

    img.savePNG(fileName);
}


void runCuda() {
    if (cameraChanged) {
        //update camera para
        iteration = 0;

        Camera &cam = renderState->camera;
        //因为视线的表示theta和phi已经求出来了，所以这里就可以直接根据theta和phi的改变来更新视线方向。
        //所以这个变量命名的不好，这里应该是 look到摄像机位置的距离向量.
        cameraPosition.x = zoom * sin(theta) * sin(phi);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * sin(theta) * cos(phi);

        cam.view = glm::normalize(cameraPosition);
        //计算u和r的时候，右手坐标系，要用view方向计算。 因为view是看向-z方向的， 在填lookat矩阵要把view方向取-作为正Z方向。
        glm::vec3 v = cam.view;
        glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
        glm::vec3 r = glm::cross(v, u);
        cam.up = glm::cross(r, v);
        cam.right = r;
        
        //lookat = position + view
        //posiiton = lookat - view. = (+ camerePosition).
        cam.position = cam.lookAt - cameraPosition;
        cameraChanged = false;
    }

    if (iteration == 0) {
        pathTraceFree();
        pathTraceInit(scene);
    }

    if (iteration < renderState->iterations) {
        uchar4* pbo_dptr = nullptr;
        ++iteration;
        cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

        int frame = 0;
        pathTrace(pbo_dptr, frame, iteration);

        cudaGLUnmapBufferObject(pbo);
    } else {
        saveImage();
        pathTraceFree();
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
}




void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        switch (key) {
        case GLFW_KEY_ESCAPE:
            saveImage();
            glfwWindowShouldClose(window);
            break;
        case GLFW_KEY_S:
            saveImage();
            break;
        case GLFW_KEY_SPACE:
            cameraChanged = true;
            renderState = &scene->state;
            Camera &camera = renderState->camera;
            camera.lookAt = ogLookAt;
            break;
        }
    }
}


void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    leftMousePressed    = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed   = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    middleMousePressed  = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}


void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
    if (xpos == xPos || ypos == yPos) { return; }

    /************************************************************************/
    /* problem                                                              */
    /* 这里需要调整以下变量改动， 来更好的理解交互    已根据自己的理解修改，work                  */
    /************************************************************************/

    if (leftMousePressed) {
        //update camera parameters
        phi -= (xpos - xPos) / static_cast<double>(width);
        theta += (ypos - yPos) / static_cast<double>(height);
        theta = std::fmax(0.001f, std::fmin(theta, PI));
        cameraChanged = true;
    }

    if (rightMousePressed) {
        zoom += (ypos - yPos) / static_cast<double>(height);
        zoom = std::fmax(0.1f, zoom);
        cameraChanged = true;
    }

    if (middleMousePressed) {
        renderState = &scene->state;
        Camera &cam = renderState->camera;
        glm::vec3 forward = cam.view;
        forward.y = 0.0f;
        forward = glm::normalize(forward);
        glm::vec3 right = cam.right;
        right.y = 0;
        right = glm::normalize(right);

        cam.lookAt -= static_cast<float>(xpos - xPos) * right * 0.01f;
        cam.lookAt += static_cast<float>(ypos - yPos) * forward * 0.01f;
        cameraChanged = true;
    }

    xPos = xpos;
    yPos = ypos;

}