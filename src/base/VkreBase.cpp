/*
 * Vulkan Image Filtering Application
 *
 * Copyright (C) 2018 by Joon Jung
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */


#include "VkreBase.h"

std::vector<const char*> VkreBase::args;

const std::string VkreBase::getAssetPath()
{
  return VK_EXAMPLE_DATA_DIR;
}

VkreBase::VkreBase(bool enableValidation)
{
  // Check for a valid asset path
  struct stat info;
  if (stat(getAssetPath().c_str(), &info) != 0)
  {
    std::cerr << "Error: Could not find asset path in " << getAssetPath() << std::endl;
    exit(-1);
  }

  settings.validation = enableValidation;

  char* numConvPtr;

  // Parse command line arguments
  for (size_t i = 0; i < args.size(); i++)
  {
    if (args[i] == std::string("-validation")) {
      settings.validation = true;
    }
    if (args[i] == std::string("-vsync")) {
      settings.vsync = true;
    }
    if ((args[i] == std::string("-f")) || (args[i] == std::string("--fullscreen"))) {
      settings.fullscreen = true;
    }
    if ((args[i] == std::string("-w")) || (args[i] == std::string("-width"))) {
      uint32_t w = strtol(args[i + 1], &numConvPtr, 10);
      if (numConvPtr != args[i + 1]) { width = w; };
    }
    if ((args[i] == std::string("-h")) || (args[i] == std::string("-height"))) {
      uint32_t h = strtol(args[i + 1], &numConvPtr, 10);
      if (numConvPtr != args[i + 1]) { height = h; };
     }
    // Benchmark
    if ((args[i] == std::string("-b")) || (args[i] == std::string("--benchmark"))) {
      benchmark.active = true;
      vks::tools::errorModeSilent = true;
    }
    // Warmup time (in seconds)
    if ((args[i] == std::string("-bw")) || (args[i] == std::string("--benchwarmup"))) {
      if (args.size() > i + 1) {
        uint32_t num = strtol(args[i + 1], &numConvPtr, 10);
        if (numConvPtr != args[i + 1]) {
          benchmark.warmup = num;
        } else {
          std::cerr << "Warmup time for benchmark mode must be specified as a number!" << std::endl;
        }
      }  
    }
    // Benchmark runtime (in seconds)
    if ((args[i] == std::string("-br")) || (args[i] == std::string("--benchruntime"))) {
      if (args.size() > i + 1) {
        uint32_t num = strtol(args[i + 1], &numConvPtr, 10);
	if (numConvPtr != args[i + 1]) {
	  benchmark.duration = num;
        }
	else {
	  std::cerr << "Benchmark run duration must be specified as a number!" << std::endl;
	}
      }
    }
    // Bench result save filename (overrides default)
    if ((args[i] == std::string("-bf")) || (args[i] == std::string("--benchfilename"))) {
      if (args.size() > i + 1) {
        if (args[i + 1][0] == '-') {
	  std::cerr << "Filename for benchmark results must not start with a hyphen!" << std::endl;
	} else {
	  benchmark.filename = args[i + 1];
        }
      }
    }
    // Output frame times to benchmark result file
    if ((args[i] == std::string("-bt")) || (args[i] == std::string("--benchframetimes"))) {
      benchmark.outputFrameTimes = true;
    }
  }
	
	initxcbConnection();
}

VkResult VkreBase::createInstance(bool enableValidation)
{
  this->settings.validation = enableValidation;

#if defined(_VALIDATION)
  this->settings.validation = true;
#endif

  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = name.c_str();
  appInfo.pEngineName = name.c_str();
  appInfo.apiVersion = VK_API_VERSION_1_0;

  std::vector<const char*> instanceExtensions = { VK_KHR_SURFACE_EXTENSION_NAME };

  // Enable XCB dependent extension
  instanceExtensions.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);

  VkInstanceCreateInfo instanceCreateInfo = {};
  instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceCreateInfo.pNext = NULL;
  instanceCreateInfo.pApplicationInfo = &appInfo;

  if (instanceExtensions.size() > 0) {
    if (settings.validation)
    {
      instanceExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    }
    instanceCreateInfo.enabledExtensionCount = (uint32_t)instanceExtensions.size();
    instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions.data();
  }
  if (settings.validation) {
    instanceCreateInfo.enabledLayerCount = vks::debug::validationLayerCount;
    instanceCreateInfo.ppEnabledLayerNames = vks::debug::validationLayerNames;
  }

  return vkCreateInstance(&instanceCreateInfo, nullptr, &instance);
}

void VkreBase::getEnabledFeatures()
{
  // over ride side door
}


void VkreBase::initVulkan()
{
  VkResult err;

  // Create the instance
  err = createInstance(settings.validation);
  if (err)
    vks::tools::exitFatal("Could not create Vulkan instance: \n" + vks::tools::errorString(err), err);
  
  if (settings.validation) {
    VkDebugReportFlagsEXT debugReportFlags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
    vks::debug::setupDebugging(instance, debugReportFlags, VK_NULL_HANDLE);
  }

  uint32_t gpuCount = 0;
  VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr));
  assert(gpuCount > 0);
  std::vector<VkPhysicalDevice> physicalDevices(gpuCount);
  err = vkEnumeratePhysicalDevices(instance, &gpuCount, physicalDevices.data());
  if (err) {
    vks::tools::exitFatal("Failed to find physical device(s) \n" + vks::tools::errorString(err), err);
  }

  // GPU selection
  uint32_t selectedDevice = 0;
  for (size_t i = 0; i < args.size(); i++) {
    if ((args[i] == std::string("-g")) || (args[i] == std::string("-gpu"))) {
      char* endptr;
      uint32_t index = strtol(args[i + 1], &endptr, 10);
      if (endptr != args[i + 1]) {
        if (index > gpuCount - 1) {
          std::cerr << "Selected device index " << index << " is out of range, setting to device 0" << std::endl;
        }
        else {
          std::cout << "Selected a Vulkan device " << index << std::endl;
          selectedDevice = index;
        }
      }
      break;
    }

    // List available GPUs
    if (args[i] == std::string("-listgpus"))
    {
      uint32_t gpuCount = 0;
      VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr));
      if (gpuCount == 0) {
        std::cerr << "No Vulkane device found." << std::endl;
      }
      else {
        std::cout << "Available Vulkane devices" << std::endl;
        std::vector<VkPhysicalDevice> devices(gpuCount);
        VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance, &gpuCount, devices.data()));
        for (uint32_t i = 0; i < gpuCount; i++) {
          VkPhysicalDeviceProperties deviceProperties;
          vkGetPhysicalDeviceProperties(devices[i], &deviceProperties);
          vkGetPhysicalDeviceProperties(devices[i], &deviceProperties);
          std::cout << "Device [" << i << "] : " << deviceProperties.deviceName << std::endl;
          std::cout << " Type: " << vks::tools::physicalDeviceTypeString(deviceProperties.deviceType) << std::endl;
          std::cout << " API: " << (deviceProperties.apiVersion >> 22) << "." << ((deviceProperties.apiVersion >> 12) & 0x3ff)
                                << "." << (deviceProperties.apiVersion & 0xfff) << std::endl;
        }
      }
    }
  }
   
  physicalDevice = physicalDevices[selectedDevice];
  
  vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
  vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &deviceMemoryProperties);

  getEnabledFeatures();

  vulkanDevice = new vks::VulkanDevice(physicalDevice);
  VkResult res = vulkanDevice->createLogicalDevice(enabledFeatures, enabledExtensions);
  if (res != VK_SUCCESS) {
    vks::tools::exitFatal("Failed to create Vulkan logicak device: \n" + vks::tools::errorString(res), res);
  }
  device = vulkanDevice->logicalDevice;

  // Get the queue
  vkGetDeviceQueue(device, vulkanDevice->queueFamilyIndices.graphics, 0, &queue);

  VkBool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice, &depthFormat);
  assert(validDepthFormat);

  swapChain.connect(instance, physicalDevice, device);

  // Create semaphores
  VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
  VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphores.presentComplete));
  VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphores.renderComplete));
  VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphores.overlayComplete));

  // Command buffer submission info is set by the derived vkre calss
  submitInfo = vks::initializers::submitInfo();
  submitInfo.pWaitDstStageMask = &submitPipelineStages;
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = &semaphores.presentComplete;
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = &semaphores.renderComplete;
}

void VkreBase::initSwapChain()
{
  swapChain.initSurface(connection, window);
}

void VkreBase::setupSwapChain()
{
  swapChain.create(&width, &height, settings.vsync);
}

void VkreBase::createCommandPool()
{
  VkCommandPoolCreateInfo cmdPoolInfo = {};
  cmdPoolInfo.sType  = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cmdPoolInfo.queueFamilyIndex = swapChain.queueNodeIndex;
  cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &cmdPool));
}

void VkreBase::setupDepthStencil()
{
  VkImageCreateInfo imageInfo = {};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.pNext = NULL;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.format = depthFormat;
  imageInfo.extent = { width, height, 1};
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  imageInfo.flags = 0;

  VkMemoryAllocateInfo memAlloc = {};
  memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memAlloc.pNext = NULL;
  memAlloc.allocationSize = 0;
  memAlloc.memoryTypeIndex = 0;

  VkImageViewCreateInfo depthStencilViewInfo = {};
  depthStencilViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  depthStencilViewInfo.pNext = NULL;
  depthStencilViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  depthStencilViewInfo.format = depthFormat;
  depthStencilViewInfo.flags = 0;
  depthStencilViewInfo.subresourceRange = {};
  depthStencilViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
  depthStencilViewInfo.subresourceRange.baseMipLevel = 0;
  depthStencilViewInfo.subresourceRange.levelCount = 1;
  depthStencilViewInfo.subresourceRange.baseArrayLayer = 0;
  depthStencilViewInfo.subresourceRange.layerCount = 1;

  VkMemoryRequirements memReqs;

  VK_CHECK_RESULT(vkCreateImage(device, &imageInfo, nullptr, &depthStencil.image));
  vkGetImageMemoryRequirements(device, depthStencil.image, &memReqs);
  memAlloc.allocationSize = memReqs.size;
  memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &depthStencil.mem));
  VK_CHECK_RESULT(vkBindImageMemory(device, depthStencil.image, depthStencil.mem, 0));
  
  depthStencilViewInfo.image = depthStencil.image;
  VK_CHECK_RESULT(vkCreateImageView(device, &depthStencilViewInfo, nullptr, &depthStencil.view));
}

void VkreBase::setupRenderPass()
{
  std::array<VkAttachmentDescription, 2> attachments = {};
  
  // Color attachments
  attachments[0].format = swapChain.colorFormat;
  attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
  attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  // Depth attachment
  attachments[1].format = depthFormat;
  attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
  attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkAttachmentReference colorReference = {};
  colorReference.attachment = 0;
  colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depthReference = {};
  depthReference.attachment = 1;
  depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpassDescription = {};
  subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpassDescription.colorAttachmentCount = 1;
  subpassDescription.pColorAttachments = &colorReference;
  subpassDescription.pDepthStencilAttachment = &depthReference;
  subpassDescription.inputAttachmentCount = 0;
  subpassDescription.pInputAttachments = nullptr;
  subpassDescription.preserveAttachmentCount = 0;
  subpassDescription.pPreserveAttachments = nullptr;
  subpassDescription.pResolveAttachments = nullptr; 

  std::array<VkSubpassDependency, 2> dependencies;

  dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
  dependencies[0].dstSubpass = 0;
  dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
  dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

  dependencies[1].srcSubpass = 0;
  dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
  dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
  dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

  VkRenderPassCreateInfo renderPassInfo = {};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
  renderPassInfo.pAttachments = attachments.data();
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpassDescription;
  renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
  renderPassInfo.pDependencies = dependencies.data();

  VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));
}

void VkreBase::createPipelineCache()
{
  VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
  pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  VK_CHECK_RESULT(vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache));
}

void VkreBase::setupFrameBuffer()
{
  VkImageView attachments[2];

  attachments[1] = depthStencil.view;

  VkFramebufferCreateInfo frameBufferCreateInfo = {};
  frameBufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  frameBufferCreateInfo.pNext = NULL;
  frameBufferCreateInfo.renderPass = renderPass;
  frameBufferCreateInfo.attachmentCount = 2;
  frameBufferCreateInfo.pAttachments = attachments;
  frameBufferCreateInfo.width = width;
  frameBufferCreateInfo.height = height;
  frameBufferCreateInfo.layers = 1;

  // Create frame buffers for every swap chain image
  frameBuffers.resize(swapChain.imageCount);
  for (uint32_t i = 0; i < frameBuffers.size(); i++) {
    attachments[0] = swapChain.buffers[i].view;
    VK_CHECK_RESULT(vkCreateFramebuffer(device, &frameBufferCreateInfo, nullptr, &frameBuffers[i]));
  }
}

void VkreBase::OnSetupUIOverlay(vks::UIOverlayCreateInfo &createInfo) {}

void VkreBase::OnUpdateUIOverlay(vks::UIOverlay *overlay) {}

void VkreBase::updateOverlay()
{
  if (!settings.overlay)
    return;

  ImGuiIO &io = ImGui::GetIO();
  
  io.DisplaySize = ImVec2((float)width, (float)height);
  io.DeltaTime = frameTimer;

  io.MousePos = ImVec2(mousePos.x, mousePos.y);
  io.MouseDown[0] = mouseButtons.left;
  io.MouseDown[1] = mouseButtons.right;

  ImGui::NewFrame();

  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
  ImGui::SetNextWindowPos(ImVec2(10, 10));
  ImGui::SetNextWindowSize(ImVec2(0, 0), ImGuiSetCond_FirstUseEver);
  ImGui::Begin("VKRE", nullptr, ImGuiWindowFlags_AlwaysAutoResize | 
                                ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
  ImGui::TextUnformatted(title.c_str());
  ImGui::TextUnformatted(deviceProperties.deviceName);
  ImGui::Text("%.2f ms/frame (%.1d fps)", (1000.0f / lastFPS), lastFPS);

  ImGui::PushItemWidth(110.0f * UIOverlay->scale);
  OnUpdateUIOverlay(UIOverlay);
  ImGui::PopItemWidth();

  ImGui::End();
  ImGui::PopStyleVar();
  ImGui::Render();

  UIOverlay->update();
}

void VkreBase::prepare()
{
  if (vulkanDevice->enableDebugMarkers) {
    vks::debugmarker::setup(device);
  }

  initSwapChain();
  createCommandPool();
  setupSwapChain();
  createCommandBuffers();
  setupDepthStencil();
  setupRenderPass();
  createPipelineCache();
  setupFrameBuffer();

  settings.overlay = settings.overlay && (!benchmark.active);
  if (settings.overlay) {
    vks::UIOverlayCreateInfo overlayCreateInfo = {};
    // Default overlay creation info
    overlayCreateInfo.device = vulkanDevice;
    overlayCreateInfo.copyQueue = queue;
    overlayCreateInfo.framebuffers = frameBuffers;
    overlayCreateInfo.colorformat = swapChain.colorFormat;
    overlayCreateInfo.width = width;
    overlayCreateInfo.height = height;
    // Virtual function to allow customizing ovelay creation
    OnSetupUIOverlay(overlayCreateInfo);
    if (overlayCreateInfo.shaders.size() == 0) {
      overlayCreateInfo.shaders = {
        loadShader(getAssetPath() + "shaders/base/uioverlay.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
        loadShader(getAssetPath() + "shaders/base/uioverlay.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT),
      };
    }
    UIOverlay = new vks::UIOverlay(overlayCreateInfo);
    updateOverlay();
  }
}

bool VkreBase::checkCommandBuffers()
{
  for (auto& cmdBuffer : drawCmdBuffers) {
    if (cmdBuffer == VK_NULL_HANDLE) {
      return false;
    }
  }
  return true;
}

VkCommandBuffer VkreBase::createCommandBuffer(VkCommandBufferLevel level, bool begin)
{
  VkCommandBuffer cmdBuffer;
  VkCommandBufferAllocateInfo cmdBufferAllocateInfo =
    vks::initializers::commandBufferAllocateInfo(cmdPool, level, 1);
  VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufferAllocateInfo, &cmdBuffer));

  // If requested, also start a new command buffer
  if (begin) {
    VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));
  }
  
  return cmdBuffer;
}

void VkreBase::createCommandBuffers()
{
  // Create one command buffer for each swap chain image
  // and reuse for rendering
  drawCmdBuffers.resize(swapChain.imageCount);

  VkCommandBufferAllocateInfo cmdBufferAllocateInfo = 
    vks::initializers::commandBufferAllocateInfo(cmdPool,
                                                 VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                                 static_cast<uint32_t>(drawCmdBuffers.size()));

  VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufferAllocateInfo, drawCmdBuffers.data()));
}

void VkreBase::destroyCommandBuffers()
{
  vkFreeCommandBuffers(device, cmdPool, static_cast<uint32_t>(drawCmdBuffers.size()),
                       drawCmdBuffers.data());
}

void VkreBase::flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue,
                                  bool free)
{
  if (commandBuffer == VK_NULL_HANDLE)
  {
    return;
  }
  VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));
  
  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
  VK_CHECK_RESULT(vkQueueWaitIdle(queue));

  if (free) {
    vkFreeCommandBuffers(device, cmdPool, 1, &commandBuffer);
  }
}

void VkreBase::prepareFrame()
{
  // Acquire the next image from the swap chain
  VkResult err = swapChain.acquireNextImage(semaphores.presentComplete, &currentBuffer);
  if ((err == VK_ERROR_OUT_OF_DATE_KHR) || (err == VK_SUBOPTIMAL_KHR)) {
    windowResize();
  }
  else {
    VK_CHECK_RESULT(err);
  }
}

void VkreBase::submitFrame()
{
  bool submitOverlay = settings.overlay && UIOverlay->visible;

  if (submitOverlay) {
    // Wait for color attachment output to finish
    // before rendering the text overlay
    VkPipelineStageFlags stageFlags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    submitInfo.pWaitDstStageMask = &stageFlags;
    
    // Wait for render complete semaphore
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &semaphores.renderComplete;
   
    // Signal ready with UI overlay complete semaphore
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &semaphores.overlayComplete;

    // Submit current UI overlay command buffer
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &UIOverlay->cmdBuffers[currentBuffer];
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

    // Reset
    submitInfo.pWaitDstStageMask = &submitPipelineStages;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &semaphores.presentComplete;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &semaphores.renderComplete;
  }

  VK_CHECK_RESULT(swapChain.queuePresent(queue, currentBuffer,
            submitOverlay ? semaphores.overlayComplete : semaphores.renderComplete));
  VK_CHECK_RESULT(vkQueueWaitIdle(queue));
}

void VkreBase::renderFrame()
{
  auto tStart = std::chrono::high_resolution_clock::now();

  if (viewUpdated) {
    viewUpdated = false;
    viewChanged();
  }

  render();
  frameCounter++;

  auto tEnd = std::chrono::high_resolution_clock::now();
  auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

  frameTimer = (float)tDiff / 1000.0f;
  camera.update(frameTimer);
  if (camera.moving()) {
    viewUpdated = true;
  }

  if (!paused) {
    timer += timerSpeed * frameTimer;
    if (timer > 1.0) {
      timer -= 1.0f;
    }
  }
  fpsTimer += (float)tDiff;
  if (fpsTimer > 1000.0f) {
    lastFPS = static_cast<uint32_t>((float)frameCounter * (1000.0f / fpsTimer));
    fpsTimer = 0.0f;
    frameCounter = 0;
  }

  updateOverlay();
}

void VkreBase::renderLoop()
{
  if (benchmark.active) {
    benchmark.run([=] { render(); }, vulkanDevice->properties);
    vkDeviceWaitIdle(device);
    if (benchmark.filename != "") {
      benchmark.saveResults();
    }
    return;
  }

  destWidth = width;
  destHeight = height;
  xcb_flush(connection);

  while(!quit) {
    auto tStart = std::chrono::high_resolution_clock::now();
  
    if (viewUpdated) {
      viewUpdated = false;
      viewChanged();
    }
    xcb_generic_event_t* event;
    while ((event = xcb_poll_for_event(connection))) {
      handleEvent(event);
      free(event);
    }

    render();
    frameCounter++;

    auto tEnd = std::chrono::high_resolution_clock::now();
    auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    frameTimer = tDiff / 1000.0f;
    camera.update(frameTimer);
    if (camera.moving()) {
      viewUpdated = true;
    }

   if (!paused) {
      timer += timerSpeed * frameTimer;
      if (timer > 1.0) {
        timer -= 1.0f;
      }
    }
    fpsTimer += (float)tDiff;
    if (fpsTimer > 1000.0f) {
      if (!settings.overlay) {
        std::string windowTitle = getWindowTitle();
        xcb_change_property(connection, XCB_PROP_MODE_REPLACE,
                            window, XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8,
                            windowTitle.size(), windowTitle.c_str());
      }
      lastFPS = (float)frameCounter * (1000.0f / fpsTimer);
      fpsTimer = 0.0f;
      frameCounter = 0;
    }
    updateOverlay();
  }

  vkDeviceWaitIdle(device);
}       

void VkreBase::handleEvent(const xcb_generic_event_t *event)
{
  switch (event->response_type & 0x7f) {
    case XCB_CLIENT_MESSAGE:
    {
      if ((*(xcb_client_message_event_t*)event).data.data32[0] ==
	  (*atom_wm_delete_window).atom) {
        quit = true;
      }
    }
    break;
    case XCB_MOTION_NOTIFY:
    {
      xcb_motion_notify_event_t *motion = (xcb_motion_notify_event_t *)event;
      handleMouseMove((int32_t)motion->event_x, (int32_t)motion->event_y);
    }
    break;
    case XCB_BUTTON_PRESS:
    {
      xcb_button_press_event_t *press = (xcb_button_press_event_t *)event;
      if (press->detail == XCB_BUTTON_INDEX_1)
        mouseButtons.left = true;
      if (press->detail == XCB_BUTTON_INDEX_2)
        mouseButtons.middle = true;
      if (press->detail == XCB_BUTTON_INDEX_3)
        mouseButtons.right = true;
    }
    break;
    case XCB_BUTTON_RELEASE:
   {
     xcb_button_press_event_t *press = (xcb_button_press_event_t *)event;
      if (press->detail == XCB_BUTTON_INDEX_1)
        mouseButtons.left = false;
      if (press->detail == XCB_BUTTON_INDEX_2)
        mouseButtons.middle = false;
      if (press->detail == XCB_BUTTON_INDEX_3)
        mouseButtons.right = false;
    }
    break; 
    case XCB_KEY_PRESS:
    {
      const xcb_key_release_event_t *keyEvent = (const xcb_key_release_event_t *)event;
      switch (keyEvent->detail)
      {
        case KEY_W:
	  camera.keys.up = true;
	  break;
	case KEY_S:
	  camera.keys.down = true;
	  break;
	case KEY_A:
	  camera.keys.left = true;
	  break;
	case KEY_D:
	  camera.keys.right = true;
	  break;
	case KEY_P:
	  paused = !paused;
	  break;
	case KEY_F1:
	  if (settings.overlay) {
	    settings.overlay = !settings.overlay;
	  }
	  break;				
      }
    }
    break;	
    case XCB_KEY_RELEASE:
    {
      const xcb_key_release_event_t *keyEvent = (const xcb_key_release_event_t *)event;
      switch (keyEvent->detail)
      {
        case KEY_W:
	  camera.keys.up = false;
	  break;
	case KEY_S:
	  camera.keys.down = false;
	  break;
	case KEY_A:
	  camera.keys.left = false;
	  break;
	case KEY_D:
	  camera.keys.right = false;
	  break;			
	case KEY_ESCAPE:
	  quit = true;
	  break;
	}
	keyPressed(keyEvent->detail);
    }
    break;
    case XCB_DESTROY_NOTIFY:
      quit = true;
      break;
    case XCB_CONFIGURE_NOTIFY:
    {
      const xcb_configure_notify_event_t *cfgEvent = (const xcb_configure_notify_event_t *)event;
      if ((prepared) && ((cfgEvent->width != width) || (cfgEvent->height != height)))
      {
        destWidth = cfgEvent->width;
	destHeight = cfgEvent->height;
	if ((destWidth > 0) && (destHeight > 0))
	{
	  windowResize();
	}
      }
    }
    break;
    default:
      break;
  }
}

void VkreBase::viewChanged() {}

void VkreBase::keyPressed(uint32_t) {}

void VkreBase::mouseMoved(double x, double y, bool & handled) {}

void VkreBase::buildCommandBuffers() {}


////////////////////////////////////////////////////////////////////////////////

std::string VkreBase::getWindowTitle()
{
	std::string device(deviceProperties.deviceName);
	std::string windowTitle;
	windowTitle = title + " - " + device;
	if (!settings.overlay) {
		windowTitle += " - " + std::to_string(frameCounter) + " fps";
	}
	return windowTitle;
}

VkPipelineShaderStageCreateInfo VkreBase::loadShader(std::string fileName, VkShaderStageFlagBits stage)
{
	VkPipelineShaderStageCreateInfo shaderStage = {};
	shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderStage.stage = stage;
	shaderStage.module = vks::tools::loadShader(fileName.c_str(), device);
	shaderStage.pName = "main"; // todo : make param
	assert(shaderStage.module != VK_NULL_HANDLE);
	shaderModules.push_back(shaderStage.module);
	return shaderStage;
}

VkreBase::~VkreBase()
{
	// Clean up Vulkan resources
	swapChain.cleanup();
	if (descriptorPool != VK_NULL_HANDLE)
	{
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
	}
	destroyCommandBuffers();
	vkDestroyRenderPass(device, renderPass, nullptr);
	for (uint32_t i = 0; i < frameBuffers.size(); i++)
	{
		vkDestroyFramebuffer(device, frameBuffers[i], nullptr);
	}

	for (auto& shaderModule : shaderModules)
	{
		vkDestroyShaderModule(device, shaderModule, nullptr);
	}
	vkDestroyImageView(device, depthStencil.view, nullptr);
	vkDestroyImage(device, depthStencil.image, nullptr);
	vkFreeMemory(device, depthStencil.mem, nullptr);

	vkDestroyPipelineCache(device, pipelineCache, nullptr);

	vkDestroyCommandPool(device, cmdPool, nullptr);

	vkDestroySemaphore(device, semaphores.presentComplete, nullptr);
	vkDestroySemaphore(device, semaphores.renderComplete, nullptr);
	vkDestroySemaphore(device, semaphores.overlayComplete, nullptr);

	if (UIOverlay) {
		delete UIOverlay;
	}

	delete vulkanDevice;

	if (settings.validation)
	{
		vks::debug::freeDebugCallback(instance);
	}

	vkDestroyInstance(instance, nullptr);

	xcb_destroy_window(connection, window);
	xcb_disconnect(connection);
}

static inline xcb_intern_atom_reply_t* intern_atom_helper(xcb_connection_t *conn, bool only_if_exists, const char *str)
{
	xcb_intern_atom_cookie_t cookie = xcb_intern_atom(conn, only_if_exists, strlen(str), str);
	return xcb_intern_atom_reply(conn, cookie, NULL);
}

// Set up a window using XCB and request event types
xcb_window_t VkreBase::setupWindow()
{
	uint32_t value_mask, value_list[32];

	window = xcb_generate_id(connection);

	value_mask = XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK;
	value_list[0] = screen->black_pixel;
	value_list[1] =
		XCB_EVENT_MASK_KEY_RELEASE |
		XCB_EVENT_MASK_KEY_PRESS |
		XCB_EVENT_MASK_EXPOSURE |
		XCB_EVENT_MASK_STRUCTURE_NOTIFY |
		XCB_EVENT_MASK_POINTER_MOTION |
		XCB_EVENT_MASK_BUTTON_PRESS |
		XCB_EVENT_MASK_BUTTON_RELEASE;

	if (settings.fullscreen)
	{
		width = destWidth = screen->width_in_pixels;
		height = destHeight = screen->height_in_pixels;
	}

	xcb_create_window(connection,
		XCB_COPY_FROM_PARENT,
		window, screen->root,
		0, 0, width, height, 0,
		XCB_WINDOW_CLASS_INPUT_OUTPUT,
		screen->root_visual,
		value_mask, value_list);

	/* Magic code that will send notification when window is destroyed */
	xcb_intern_atom_reply_t* reply = intern_atom_helper(connection, true, "WM_PROTOCOLS");
	atom_wm_delete_window = intern_atom_helper(connection, false, "WM_DELETE_WINDOW");

	xcb_change_property(connection, XCB_PROP_MODE_REPLACE,
		window, (*reply).atom, 4, 32, 1,
		&(*atom_wm_delete_window).atom);

	std::string windowTitle = getWindowTitle();
	xcb_change_property(connection, XCB_PROP_MODE_REPLACE,
		window, XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8,
		title.size(), windowTitle.c_str());

	free(reply);

	if (settings.fullscreen)
	{
		xcb_intern_atom_reply_t *atom_wm_state = intern_atom_helper(connection, false, "_NET_WM_STATE");
		xcb_intern_atom_reply_t *atom_wm_fullscreen = intern_atom_helper(connection, false, "_NET_WM_STATE_FULLSCREEN");
		xcb_change_property(connection,
				XCB_PROP_MODE_REPLACE,
				window, atom_wm_state->atom,
				XCB_ATOM_ATOM, 32, 1,
				&(atom_wm_fullscreen->atom));
		free(atom_wm_fullscreen);
		free(atom_wm_state);
	}	

	xcb_map_window(connection, window);

	return(window);
}

// Initialize XCB connection
void VkreBase::initxcbConnection()
{
	const xcb_setup_t *setup;
	xcb_screen_iterator_t iter;
	int scr;

	connection = xcb_connect(NULL, &scr);
	if (connection == NULL) {
		printf("Could not find a compatible Vulkan ICD!\n");
		fflush(stdout);
		exit(1);
	}

	setup = xcb_get_setup(connection);
	iter = xcb_setup_roots_iterator(setup);
	while (scr-- > 0)
		xcb_screen_next(&iter);
	screen = iter.data;
}

void VkreBase::windowResize()
{
	if (!prepared)
	{
		return;
	}
	prepared = false;

	// Ensure all operations on the device have been finished before destroying resources
	vkDeviceWaitIdle(device);

	// Recreate swap chain
	width = destWidth;
	height = destHeight;
	setupSwapChain();

	// Recreate the frame buffers
	vkDestroyImageView(device, depthStencil.view, nullptr);
	vkDestroyImage(device, depthStencil.image, nullptr);
	vkFreeMemory(device, depthStencil.mem, nullptr);
	setupDepthStencil();	
	for (uint32_t i = 0; i < frameBuffers.size(); i++) {
		vkDestroyFramebuffer(device, frameBuffers[i], nullptr);
	}
	setupFrameBuffer();

	// Command buffers need to be recreated as they may store
	// references to the recreated frame buffer
	destroyCommandBuffers();
	createCommandBuffers();
	buildCommandBuffers();

	vkDeviceWaitIdle(device);

	if (settings.overlay) {
		UIOverlay->resize(width, height, frameBuffers);
	}

	camera.updateAspectRatio((float)width / (float)height);

	// Notify derived class
	windowResized();
	viewChanged();

	prepared = true;
}

void VkreBase::handleMouseMove(int32_t x, int32_t y)
{
	int32_t dx = (int32_t)mousePos.x - x;
	int32_t dy = (int32_t)mousePos.y - y;

	bool handled = false;

	if (settings.overlay) {
		ImGuiIO& io = ImGui::GetIO();
		handled = io.WantCaptureMouse;
	}
	mouseMoved((float)x, (float)y, handled);

	if (handled) {
		mousePos = glm::vec2((float)x, (float)y);
		return;
	}

	if (mouseButtons.left) {
		rotation.x += dy * 1.25f * rotationSpeed;
		rotation.y -= dx * 1.25f * rotationSpeed;
		camera.rotate(glm::vec3(dy * camera.rotationSpeed, -dx * camera.rotationSpeed, 0.0f));
		viewUpdated = true;
	}
	if (mouseButtons.right) {
		zoom += dy * .005f * zoomSpeed;
		camera.translate(glm::vec3(-0.0f, 0.0f, dy * .005f * zoomSpeed));
		viewUpdated = true;
	}
	if (mouseButtons.middle) {
		cameraPos.x -= dx * 0.01f;
		cameraPos.y -= dy * 0.01f;
		camera.translate(glm::vec3(-dx * 0.01f, -dy * 0.01f, 0.0f));
		viewUpdated = true;
	}
	mousePos = glm::vec2((float)x, (float)y);
}

void VkreBase::windowResized()
{
	// Can be overriden in derived class
}

