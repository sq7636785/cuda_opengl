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
    const char* sceneFile = "cornell_bunny.txt";
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
    /* ����ҲӦ���ٵ�һ�£����԰�������뷨��һ��
        Դ���phi�Ǻ�z��ļнǣ� theta�Ǻ�y��ļн�
        �����theta phi ������Ľ���ȫ��һ�׵ġ�
        ��Ҫ��ϸ������£� ����ȫ���Լ����뷨��һ��
    */
    /************************************************************************/
    
    //�Ѹ�����
    //phi������Z��ļнǡ�
    //��������phi��theta������ķ��������������߷���
    //����focalLength����lookat�������λ�õľ�������

    
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
        //��Ϊ���ߵı�ʾtheta��phi�Ѿ�������ˣ���������Ϳ���ֱ�Ӹ���theta��phi�ĸı����������߷���
        //����������������Ĳ��ã�����Ӧ���� look�������λ�õľ�������.
        cameraPosition.x = zoom * sin(theta) * sin(phi);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * sin(theta) * cos(phi);

        cam.view = glm::normalize(cameraPosition);
        //����u��r��ʱ����������ϵ��Ҫ��view������㡣 ��Ϊview�ǿ���-z����ģ� ����lookat����Ҫ��view����ȡ-��Ϊ��Z����
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
    /* ������Ҫ�������±����Ķ��� �����õ���⽻��    �Ѹ����Լ�������޸ģ�work                  */
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