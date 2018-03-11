/*
 * Vulkan Image Filtering Application
 *
 * Copyright (C) 2018 by Joon Jung
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#pragma once

#include <xcb/xcb.h>

#include <iostream>
#include <chrono>
#include <sys/stat.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <string>
#include <array>
#include <numeric>

#include "vulkan/vulkan.h"

#include "keycodes.hpp"
#include "VulkanTools.h"
#include "VulkanDebug.h"
#include "VulkanUIOverlay.h"

#include "VulkanInitializers.hpp"
#include "VulkanDevice.hpp"
#include "VulkanSwapChain.hpp"
#include "camera.hpp"
#include "benchmark.hpp"

class VkreBase
{
public:
  static std::vector<const char*> args;

  bool prepared = false;
  uint32_t width = 1280;
  uint32_t height = 720;

  float frameTimer = 1.0f;

  vks::Benchmark benchmark;

  vks::VulkanDevice *vulkanDevice;

  struct Settings {
    bool validation = false;
    bool fullscreen = false;
    bool vsync = false; // vsync enforced on the swapchain?
    bool overlay = false; // UI ovelay enable?
  } settings;

  VkClearColorValue defaultClearColor ={{0.025f, 0.025f, 0.025f, 1.0f}};
  
  float zoom = 0;

  float timer = 0.0f;
  float timerSpeed = 0.25f;

  float rotationSpeed = 1.0f;
  float zoomSpeed = 1.0f;

  bool paused = false;

  Camera camera;
  glm::vec3 rotation = glm::vec3();
  glm::vec3 cameraPos = glm::vec3();
  glm::vec2 mousePos;

  std::string title = "Vulkan Render Engine";
  std::string name = "vkre";

  struct {
    VkImage image;
    VkDeviceMemory mem;
    VkImageView view;
  } depthStencil;

  struct {
    glm::vec2 axisLeft = glm::vec2(0.0f);
    glm::vec2 axisRight = glm::vec2(0.0f);
  } gamePadState;

  struct {
    bool left = false;
    bool right = false;
    bool middle = false;
  } mouseButtons;

  bool quit = false;
  xcb_connection_t *connection;
  xcb_screen_t *screen;
  xcb_window_t window;
  xcb_intern_atom_reply_t *atom_wm_delete_window;

  const std::string getAssetPath();

  VkreBase(bool enableValidation);
  virtual ~VkreBase();


  virtual VkResult createInstance(bool enableValidation);
  void getEnabledFeatures();
  void initVulkan();

  void initSwapChain();
  void setupSwapChain();
  void createCommandPool();
  void setupDepthStencil();
  void setupRenderPass();
  void createPipelineCache();

  void OnSetupUIOverlay(vks::UIOverlayCreateInfo &createInfo);
  void OnUpdateUIOverlay(vks::UIOverlay *overlay);

  void updateOverlay();

  void prepare();

  bool checkCommandBuffers();
  void createCommandBuffers();
  VkCommandBuffer createCommandBuffer(VkCommandBufferLevel level, bool begin);
  void destroyCommandBuffers();
  void flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, bool free);

  void prepareFrame();
  void submitFrame();

  // Render one frame of a render loop on platforms that sync rendering
  void renderFrame();
	
  // Start the main render loop
  void renderLoop();

  xcb_window_t setupWindow();
  void initxcbConnection();
  void handleEvent(const xcb_generic_event_t *event);

  // Pure virtual render function (override in derived class)
  virtual void render() = 0;
  // Called when view change occurs
  // Can be overriden in derived class to e.g. update uniform buffers 
  // Containing view dependant matrices
  virtual void viewChanged();

  virtual void keyPressed(uint32_t);
  virtual void mouseMoved(double x, double y, bool &handled);
  // Called when the window has been resized
  // Can be overriden in derived class to recreate or rebuild resources attached to the frame buffer / swapchain
  virtual void windowResized();
 
  // Pure virtual function to be overriden by the dervice class
  // Called in case of an event where e.g. the framebuffer has to be rebuild and thus
  // all command buffers that may reference this
  virtual void buildCommandBuffers();

  // Create framebuffers for all requested swap chain images
  // Can be overriden in derived class to setup a custom framebuffer (e.g. for MSAA)
  virtual void setupFrameBuffer();

  // Load a SPIR-V shader
  VkPipelineShaderStageCreateInfo loadShader(std::string fileName, VkShaderStageFlagBits stage);
	
protected:
  uint32_t frameCounter = 0;
  uint32_t lastFPS = 0;

  VkInstance instance;

  // Physical device related
  VkPhysicalDevice physicalDevice;
  VkPhysicalDeviceProperties deviceProperties;
  VkPhysicalDeviceFeatures deviceFeatures;
  VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
  VkPhysicalDeviceFeatures enabledFeatures{};
  std::vector<const char*> enabledExtensions;

  // Logical device
  VkDevice device;

  VkQueue queue;

  VkFormat depthFormat;
  VkCommandPool cmdPool;
  VkPipelineStageFlags submitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkSubmitInfo submitInfo;
  std::vector<VkCommandBuffer> drawCmdBuffers;

  VkRenderPass renderPass;
  std::vector<VkFramebuffer>frameBuffers;
  uint32_t currentBuffer = 0;
  
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  std::vector<VkShaderModule> shaderModules;
  VkPipelineCache pipelineCache;
  
  VulkanSwapChain swapChain;

  struct {
    VkSemaphore presentComplete; // swap chain image presentation
    VkSemaphore renderComplete;  // command buffer submission and execution
    VkSemaphore overlayComplete;  // ui ovelay submission and execution
  } semaphores;


private:
  float fpsTimer = 0.0f;
  bool viewUpdated = false;
  bool resizing = false;

  vks::UIOverlay *UIOverlay = nullptr;

  uint32_t destWidth;
  uint32_t destHeight;

  std::string getWindowTitle();
  void windowResize();
  void handleMouseMove(int32_t x, int32_t y);  

public: 


 };


