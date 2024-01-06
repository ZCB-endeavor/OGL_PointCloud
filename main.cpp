#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "stb_image.h"
#include "shader_m.h"
#include "camera.h"

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <vector_types.h>
#include <cuda_runtime.h>

#include "PointCloud.cuh"
#include "popl.hpp"

void framebuffer_size_callback(GLFWwindow *window, int width, int height);

void mouse_callback(GLFWwindow *window, double xpos, double ypos);

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);

void processInput(GLFWwindow *window);

void dataLoader(std::string path, std::vector<cv::Mat> &rgbVec, std::vector<cv::Mat> &depthVec,
                std::vector<cv::Mat> &intrinsicVec, std::vector<cv::Mat> &extrinsicVec,
                std::vector<cv::Mat> &extrinsicInvVec, std::vector<cv::Mat> &distVec,
                int img_num);

void getCoord(std::vector<Eigen::Vector3f> worldPoint, Eigen::Vector3f &maxCoord, Eigen::Vector3f &minCoord);

void getCenter(Eigen::Vector3f maxCoord, Eigen::Vector3f minCoord, Eigen::Vector3f &center);

void getFactor(Eigen::Vector3f maxCoord, Eigen::Vector3f minCoord, float &factor);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 0.8f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;    // time between current frame and last frame
float lastFrame = 0.0f;

bool mouseLeftButtonPress = false;
bool mouseRightButtonPress = false;

int main(int argc, char *argv[]) {
    popl::OptionParser op("Allowed Options");
    std::string input_path, output_path;
    int img_num, width, height;
    bool save_point_cloud;
    auto help = op.add<popl::Switch>("h", "help", "help message");
    auto disp_path_op = op.add<popl::Value<std::string>>("i", "input", "input image and param file path", "../datasets", &input_path);
    auto depth_path_op = op.add<popl::Value<std::string>>("o", "output", "output point cloud file path", "../datasets", &output_path);
    auto image_num_op = op.add<popl::Value<int>>("n", "image_num", "image number", 4, &img_num);
    auto width_op = op.add<popl::Value<int>>("", "width", "image width", 1920, &width);
    auto height_op = op.add<popl::Value<int>>("", "height", "image height", 1080, &height);
    auto save_op = op.add<popl::Value<bool>>("", "save", "save point cloud", false, &save_point_cloud);

    try {
        op.parse(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }
    if (help->is_set()) {
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }
    input_path = input_path.back() == '/' ? input_path : input_path + "/";
    output_path = output_path.back() == '/' ? output_path : output_path + "/";

    std::vector<cv::Mat> rgbVec, depthVec;
    std::vector<cv::Mat> intrinsicVec, extrinsicVec, extrinsicInvVec, distVec;
    std::vector<Eigen::Vector3f> pw, color_pw;
    std::vector<std::vector<Eigen::Vector3f>> all_pix, all_color_pix;

    // 数据预分配空间
    uchar3 *rgb;
    ushort1 *depth;
    float *intrinsic, *extrinsic;
    float3 *point, *color;
    cudaMalloc((void **) &rgb, width * height * img_num * sizeof(uchar3));
    cudaMalloc((void **) &depth, width * height * img_num * sizeof(ushort1));
    cudaMalloc((void **) &intrinsic, 9 * img_num * sizeof(float));
    cudaMalloc((void **) &extrinsic, 16 * img_num * sizeof(float));
    cudaMalloc((void **) &point, width * height * img_num * sizeof(float3));
    cudaMalloc((void **) &color, width * height * img_num * sizeof(float3));

    // 加载数据
    dataLoader(input_path, rgbVec, depthVec, intrinsicVec, extrinsicVec, extrinsicInvVec, distVec, img_num);
    for (int i = 0; i < img_num; i++) {
        cudaMemcpy(rgb + i * (width * height), rgbVec[i].data,
                   width * height * sizeof(uchar3), cudaMemcpyHostToDevice);
        cudaMemcpy(depth + i * (width * height), depthVec[i].data,
                   width * height * sizeof(ushort1), cudaMemcpyHostToDevice);
        cudaMemcpy(intrinsic + i * 9, intrinsicVec[i].data, 9 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(extrinsic + i * 16, extrinsicInvVec[i].data, 16 * sizeof(float), cudaMemcpyHostToDevice);
    }

    // 点云拼接
    cuUV2World(rgb, depth, width, height, img_num, intrinsic, extrinsic, point, color);

    // 显存下载到内存
    auto *point_cpu = (float3 *) malloc(width * height * img_num * sizeof(float3));
    auto *color_cpu = (float3 *) malloc(width * height * img_num * sizeof(float3));
    cudaMemcpy(point_cpu, point, width * height * img_num * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(color_cpu, color, width * height * img_num * sizeof(float3), cudaMemcpyDeviceToHost);

    for (int i = 0; i < width * height * img_num; i++) {
        if (color_cpu[i].x != 0 && color_cpu[i].y != 0 && color_cpu[i].z != 0) {
            pw.push_back(Eigen::Vector3f(point_cpu[i].x, point_cpu[i].y, point_cpu[i].z));
            color_pw.push_back(Eigen::Vector3f(color_cpu[i].x, color_cpu[i].y, color_cpu[i].z));
        }
    }
    std::cout << "pw.size(): " << pw.size() << std::endl;

    Eigen::Vector3f maxCoord, minCoord, center;
    float factor = 0;
    getCoord(pw, maxCoord, minCoord);
    getCenter(maxCoord, minCoord, center);
    getFactor(maxCoord, minCoord, factor);
    for (auto &i: pw) {
        i = (i - center) * factor;
    }

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "PointCloud", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed initial GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        std::cout << "Failed initial GLAD" << std::endl;
        return -1;
    }
    glEnable(GL_DEPTH_TEST);

    // OpenGL着色器编译
    Shader ourShader("../shaders/camera.vs", "../shaders/camera.fs");
    std::vector<GLfloat> vertices;
    for (int i = 0; i < pw.size(); i++) {
        vertices.push_back(pw[i].x());
        vertices.push_back(pw[i].y());
        vertices.push_back(pw[i].z());
        vertices.push_back(color_pw[i].x() / 255.f);
        vertices.push_back(color_pw[i].y() / 255.f);
        vertices.push_back(color_pw[i].z() / 255.f);
    }

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertices[0]), &vertices[0], GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *) 0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *) (3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // imgui
    const char *glsl_version = "#version 450";
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void) io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    while (!glfwWindowShouldClose(window)) {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // activate shader
        ourShader.use();
        glBindVertexArray(VAO);

        // pass projection matrix to shader (note that in this case it could change every frame)
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float) SCR_WIDTH / (float) SCR_HEIGHT, 0.1f,
                                                100.0f);
        ourShader.setMat4("projection", projection);

        // camera/view transformation
        glm::mat4 view = camera.GetViewMatrix();
        ourShader.setMat4("view", view);

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::rotate(model, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        ourShader.setMat4("model", model);

        glPointSize(1);
        glDrawArrays(GL_POINTS, 0, vertices.size());

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        {
            ImGui::Begin("Debug");
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();

    if (save_point_cloud) {
        int num = pw.size();
        std::ofstream ply(output_path + "res.ply");
        ply << "ply" << '\n'
            << "format ascii 1.0" << '\n'
            << "element vertex " << num << '\n'
            << "property float x" << '\n'
            << "property float y" << '\n'
            << "property float z" << '\n'
            << "property uchar red" << '\n'
            << "property uchar green" << '\n'
            << "property uchar blue" << '\n'
            << "end_header" << '\n';

        for (int i = 0; i < pw.size(); i++)
        {
            ply << pw[i].x() << "  " << pw[i].y() << "  " << pw[i].z() << "  " << color_pw[i].x() << "  " << color_pw[i].y() << "  " << color_pw[i].z() << "\n";
        }
        ply.close();
        std::cout << "save point cloud number: " << num << std::endl;
    }

    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}


// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow *window, double xposIn, double yposIn) {
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    if (mouseLeftButtonPress) {
        camera.ProcessMouseMovement(xoffset, yoffset);
    }
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods) {
    double xPos, yPos;
    if (action == GLFW_PRESS) {
        switch (button) {
            case GLFW_MOUSE_BUTTON_LEFT:
                mouseLeftButtonPress = true;
                mouseRightButtonPress = false;
                break;
            case GLFW_MOUSE_BUTTON_RIGHT:
                mouseRightButtonPress = true;
                mouseLeftButtonPress = false;
                break;
            default:
                break;
        }
    } else {
        mouseLeftButtonPress = false;
        mouseRightButtonPress = false;
    }
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

void dataLoader(std::string path, std::vector<cv::Mat> &rgbVec, std::vector<cv::Mat> &depthVec,
                std::vector<cv::Mat> &intrinsicVec, std::vector<cv::Mat> &extrinsicVec,
                std::vector<cv::Mat> &extrinsicInvVec, std::vector<cv::Mat> &distVec,
                int img_num) {
    for (int i = 0; i < img_num; i++) {
        std::cout << "load " << i << " image..." << std::endl;
        std::stringstream ss;
        ss << path << "color/" << std::setw(4) << std::setfill('0') << i << ".png";

        std::stringstream ss2;
        ss2 << path << "depth/" << std::setw(4) << std::setfill('0') << i << ".png";
        cv::Mat rgb = cv::imread(ss.str(), cv::IMREAD_COLOR);
        cv::Mat depth = cv::imread(ss2.str(), cv::IMREAD_UNCHANGED);

        std::stringstream ss3;
        ss3 << path << "param/" << std::setw(4) << std::setfill('0') << i << ".xml";
        cv::FileStorage xml(ss3.str(), cv::FileStorage::READ);
        cv::Mat intrinsic, extrinsic, extrinsicInv, dist, intrinsicIr, distIr;
        xml["IntrinsicColor"] >> intrinsic;
        xml["ExtrinsicWorld"] >> extrinsic;
        xml["DistCoeffColor"] >> dist;
        intrinsic.convertTo(intrinsic, CV_32FC1);
        extrinsic.convertTo(extrinsic, CV_32FC1);
        dist.convertTo(dist, CV_32FC1);

        // 去畸变
        cv::Mat map1, map2;
        initUndistortRectifyMap(intrinsic, dist, cv::Mat(), intrinsic, rgb.size(), CV_32FC1, map1, map2);
        cv::Mat res_color, res_depth;
        cv::remap(rgb, res_color, map1, map2, cv::INTER_NEAREST);
        cv::remap(depth, res_depth, map1, map2, cv::INTER_NEAREST);

        rgbVec.push_back(res_color);
        depthVec.push_back(res_depth);

        intrinsicVec.push_back(intrinsic);
        extrinsicVec.push_back(extrinsic);
        distVec.push_back(dist);
        extrinsicInvVec.push_back(extrinsic.inv());
        xml.release();
    }
}

void getCoord(std::vector<Eigen::Vector3f> worldPoint, Eigen::Vector3f &maxCoord, Eigen::Vector3f &minCoord) {
    maxCoord.x() = worldPoint[0].x();
    maxCoord.y() = worldPoint[0].y();
    maxCoord.z() = worldPoint[0].z();
    minCoord = maxCoord;

    for (int i = 1; i < worldPoint.size(); i++) {
        if (maxCoord.x() < worldPoint[i].x()) {
            maxCoord.x() = worldPoint[i].x();
        }
        if (maxCoord.y() < worldPoint[i].y()) {
            maxCoord.y() = worldPoint[i].y();
        }
        if (maxCoord.z() < worldPoint[i].z()) {
            maxCoord.z() = worldPoint[i].z();
        }

        if (minCoord.x() > worldPoint[i].x()) {
            minCoord.x() = worldPoint[i].x();
        }
        if (minCoord.y() > worldPoint[i].y()) {
            minCoord.y() = worldPoint[i].y();
        }
        if (minCoord.z() > worldPoint[i].z()) {
            minCoord.z() = worldPoint[i].z();
        }
    }
}

void getCenter(Eigen::Vector3f maxCoord, Eigen::Vector3f minCoord, Eigen::Vector3f &center) {
    center.x() = (maxCoord.x() + minCoord.x()) / 2;
    center.y() = (maxCoord.y() + minCoord.y()) / 2;
    center.z() = (maxCoord.z() + minCoord.z()) / 2;
}

void getFactor(Eigen::Vector3f maxCoord, Eigen::Vector3f minCoord, float &factor) {
    float max = 0;
    if (max <= maxCoord.x()) {
        max = maxCoord.x();
    }
    if (max <= maxCoord.y()) {
        max = maxCoord.y();
    }
    if (max <= maxCoord.z()) {
        max = maxCoord.z();
    }
    factor = 0.5 / max;
}