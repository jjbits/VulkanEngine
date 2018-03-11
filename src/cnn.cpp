/*
 * Vulkan Image Filtering Application
 *
 * Copyright (C) 2018 by Joon Jung
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>

#include <vulkan/vulkan.h>
#include "base/VkreBase.h"
#include "base/VulkanTexture.hpp"
#include "base/VulkanBuffer.hpp"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

struct Vertex {
  float pos[3];
  float uv[2];
};

class VulkanCNN : public VkreBase
{
private:
  vks::Texture2D textureColorMap;
  vks::Texture2D textureComputeTarget;

public:
  struct {
    VkPipelineVertexInputStateCreateInfo inputState;
    std::vector<VkVertexInputBindingDescription> bindingDescriptions;
    std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
  } vertices;

  struct {
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSet descriptorSetPreCompute;
    VkDescriptorSet descriptorSetPostCompute;
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
  } graphics;

  struct Compute {
    VkQueue queue;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    VkFence fence;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSet descriptorSet;
    VkPipelineLayout pipelineLayout;
    std::vector<VkPipeline> pipelines;
    int32_t pipelineIndex = 2;
    uint32_t queueFamilyIndex;
  } compute;

  vks::Buffer vertexBuffer;
  vks::Buffer indexBuffer;
  uint32_t indexCount;

  vks::Buffer uniformBufferVS;

  struct {
    glm::mat4 projection;
    glm::mat4 model;
  } uboVS;

  int vertexBufferSize;

  std::vector<std::string> shaderNames;

  VulkanCNN() : VkreBase(ENABLE_VALIDATION)
  {
    zoom = -2.0f;
    title = "CNN Test";
    settings.overlay = true;
  }

  ~VulkanCNN()
  {
    vkDestroyPipeline(device, graphics.pipeline, nullptr);
    vkDestroyPipelineLayout(device, graphics.pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, graphics.descriptorSetLayout, nullptr);
 
    for (auto &pipeline : compute.pipelines) {
      vkDestroyPipeline(device, pipeline, nullptr);
    }
    vkDestroyPipelineLayout(device, compute.pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, compute.descriptorSetLayout, nullptr);
    vkDestroyFence(device, compute.fence, nullptr);
    vkDestroyCommandPool(device, compute.commandPool, nullptr);

    vertexBuffer.destroy();
    indexBuffer.destroy();
    uniformBufferVS.destroy();

    textureColorMap.destroy();
    textureComputeTarget.destroy();    
  }

  void loadAssets()
  {
    textureColorMap.STBloadFromFile(
      getAssetPath() + "textures/ppotto.jpg",
      VK_FORMAT_R8G8B8A8_UNORM,
      vulkanDevice,
      queue,
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
      VK_IMAGE_LAYOUT_GENERAL);

  }

  // setup vertices for a single uv-mapped quad
  void generateQuad()
  {
    std::vector<Vertex> vertices = { { { 1.0f,  1.0f, 0.0f}, {1.0f, 1.0f} },
                                     { {-1.0f,  1.0f, 0.0f}, {0.0f, 1.0f} },
                                     { {-1.0f, -1.0f, 0.0f}, {0.0f, 0.0f} },
                                     { { 1.0f, -1.0f, 0.0f}, {1.0f, 0.0f} } };

    std::vector<uint32_t> indices = {0, 1, 2, 2, 3, 0};
    indexCount = static_cast<uint32_t>(indices.size());
    // Vertex Buffer
    VK_CHECK_RESULT(vulkanDevice->createBuffer(
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      &vertexBuffer,
      vertices.size() * sizeof(Vertex),
      vertices.data()));
    // Index buffer
    VK_CHECK_RESULT(vulkanDevice->createBuffer(
      VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      &indexBuffer,
      indices.size() * sizeof(uint32_t),
      indices.data()));
  }

  void setupVertexDescriptions()
  {
    // Binding description
    vertices.bindingDescriptions.resize(1);
    vertices.bindingDescriptions[0] = vks::initializers::vertexInputBindingDescription(
                                        VERTEX_BUFFER_BIND_ID,
                                        sizeof(Vertex),
                                        VK_VERTEX_INPUT_RATE_VERTEX);
    // Attribute descriptions
    // Describes memory layout and shader positions
    vertices.attributeDescriptions.resize(2);
    // Location 0 : Position
    vertices.attributeDescriptions[0] = vks::initializers::vertexInputAttributeDescription(
                                          VERTEX_BUFFER_BIND_ID,
                                          0,
                                          VK_FORMAT_R32G32B32_SFLOAT,
                                          offsetof(Vertex, pos));
     // Location 1 : Texture coordinates
    vertices.attributeDescriptions[1] = vks::initializers::vertexInputAttributeDescription(
                                          VERTEX_BUFFER_BIND_ID,
                                          1,
                                          VK_FORMAT_R32G32_SFLOAT,
                                          offsetof(Vertex, uv));
    
    // Assign to vertex buffer
    vertices.inputState = vks::initializers::pipelineVertexInputStateCreateInfo();
    vertices.inputState.vertexBindingDescriptionCount = vertices.bindingDescriptions.size();
    vertices.inputState.pVertexBindingDescriptions = vertices.bindingDescriptions.data();
    vertices.inputState.vertexAttributeDescriptionCount = vertices.attributeDescriptions.size();
    vertices.inputState.pVertexAttributeDescriptions = vertices.attributeDescriptions.data();    
  }

  void updateUniformBuffers()
  {
    uboVS.projection = glm::perspective(glm::radians(60.0f), (float)width * 0.5f / (float)height, 0.1f, 256.0f);
    glm::mat4 viewMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, zoom));

    uboVS.model = viewMatrix * glm::translate(glm::mat4(1.0f), cameraPos);
    uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
    uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
    uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

    memcpy(uniformBufferVS.mapped, &uboVS, sizeof(uboVS));
  }

  void prepareUniformBuffers()
  {
    VK_CHECK_RESULT(vulkanDevice->createBuffer(
                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      &uniformBufferVS,
                      sizeof(uboVS)));

    VK_CHECK_RESULT(uniformBufferVS.map());
   
    updateUniformBuffers();
  }

  void prepareTextureTarget(vks::Texture *tex, uint32_t width, uint32_t height, VkFormat format)
  {
    VkFormatProperties formatProperties;

    // Get device properties for the requested texture format
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProperties);
    // Check i requested image format supports image storage operations
    assert(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);

    tex->width = width;
    tex->height = height;

    VkImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = format;
    imageCreateInfo.extent = { width, height, 1 };
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    // Image will be sampled in the fragment shader and used as storage target in the compute shader
    imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    imageCreateInfo.flags = 0;
    // Ownership does not needed to be exclusively transferred
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK_RESULT(vkCreateImage(device, &imageCreateInfo, nullptr, &tex->image));

    VkMemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, tex->image, &memReqs);
    memAllocInfo.allocationSize = memReqs.size;
    memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(device, &memAllocInfo, nullptr, &tex->deviceMemory));
    VK_CHECK_RESULT(vkBindImageMemory(device, tex->image, tex->deviceMemory, 0));

    VkCommandBuffer layoutCmd = VkreBase::createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    tex->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    vks::tools::setImageLayout(layoutCmd, tex->image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, 
                               tex->imageLayout);
    VkreBase::flushCommandBuffer(layoutCmd, queue, true);

    // Sampler
    VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler.addressModeV = sampler.addressModeU;
    sampler.addressModeW = sampler.addressModeU;
    sampler.mipLodBias = 0.0f;
    sampler.maxAnisotropy = 1.0f;
    sampler.compareOp = VK_COMPARE_OP_NEVER;
    sampler.minLod = 0.0f;
    sampler.maxLod = 0.0f;
    sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &tex->sampler));

    // Image view
    VkImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
    view.image = VK_NULL_HANDLE;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = format;
    view.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G,
                        VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
    view.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    view.image = tex->image;
    VK_CHECK_RESULT(vkCreateImageView(device, &view, nullptr, &tex->view));

    // Initialize a descriptor for later use
    tex->descriptor.imageLayout = tex->imageLayout;
    tex->descriptor.imageView = tex->view;
    tex->descriptor.sampler = tex->sampler;
    tex->device = vulkanDevice;    
  }
  
  void setupDescriptorSetLayout()
  {
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = 
    { 
      // Binding 0: Vertex shader uniform buffer
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                    VK_SHADER_STAGE_VERTEX_BIT,
                                                    0),
      // Binding 1: Fragment shader image sampler
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                    VK_SHADER_STAGE_FRAGMENT_BIT,
                                                    1)
    };
    VkDescriptorSetLayoutCreateInfo descriptorLayout = 
      vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(),
                                                       setLayoutBindings.size());
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &graphics.descriptorSetLayout));

    VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
      vks::initializers::pipelineLayoutCreateInfo(&graphics.descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &graphics.pipelineLayout));
  }

  void preparePipelines()
  {
    // Rendering pipeline  

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = 
      vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                                              0,
                                                              VK_FALSE);

    VkPipelineRasterizationStateCreateInfo rasterizationState =
      vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL,
                                                              VK_CULL_MODE_NONE,
                                                              VK_FRONT_FACE_COUNTER_CLOCKWISE,
                                                              0);

    VkPipelineColorBlendAttachmentState blendAttachmentState = 
      vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
    VkPipelineColorBlendStateCreateInfo colorBlendState =
      vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);

    VkPipelineDepthStencilStateCreateInfo depthStencilState =
      vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, 
                                                             VK_TRUE,
                                                             VK_COMPARE_OP_LESS_OR_EQUAL);
  
    VkPipelineViewportStateCreateInfo viewportState = 
      vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);

    VkPipelineMultisampleStateCreateInfo multisampleState =
      vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);

    std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT,
                                                        VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState =
      vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables.data(),
                                                        dynamicStateEnables.size(),
                                                        0);
    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;
    shaderStages[0] = loadShader(getAssetPath() + "shaders/computeshader/texture.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getAssetPath() + "shaders/computeshader/texture.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);
    
    VkGraphicsPipelineCreateInfo pipelineCreateInfo =
      vks::initializers::pipelineCreateInfo(graphics.pipelineLayout, renderPass, 0);
    pipelineCreateInfo.pVertexInputState = &vertices.inputState;
    pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
    pipelineCreateInfo.pRasterizationState = &rasterizationState;
    pipelineCreateInfo.pColorBlendState = &colorBlendState;
    pipelineCreateInfo.pMultisampleState = &multisampleState;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pDepthStencilState = &depthStencilState;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.stageCount = shaderStages.size();
    pipelineCreateInfo.pStages = shaderStages.data();
    pipelineCreateInfo.renderPass = renderPass;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &graphics.pipeline));
  }

  void setupDescriptorPool()
  {
    std::vector<VkDescriptorPoolSize> poolSizes = 
    { vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2),
      // Graphics pipeline
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2),
      // Compute pipeline reading
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1),
      // Compute pipeline reading and writing
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2) };

    VkDescriptorPoolCreateInfo descriptorPoolInfo = 
      vks::initializers::descriptorPoolCreateInfo(poolSizes.size(), poolSizes.data(), 3);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
  }

  void setupDescriptorSet()
  {
    VkDescriptorSetAllocateInfo allocInfo = 
      vks::initializers::descriptorSetAllocateInfo(descriptorPool, &graphics.descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &graphics.descriptorSetPostCompute));

    std::vector< VkWriteDescriptorSet> writeDescriptorSets =
    {
      // Binding 0: Vertex shader uniform buffer
      vks::initializers::writeDescriptorSet(graphics.descriptorSetPostCompute,
                                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                            0,
                                            &uniformBufferVS.descriptor),
      // Binding 1: Fragment shader texture sampler
      vks::initializers::writeDescriptorSet(graphics.descriptorSetPostCompute,
                                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                            1,
                                            &textureComputeTarget.descriptor)
    };
    vkUpdateDescriptorSets(device, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);

    // Base image
    allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool,
                                                          &graphics.descriptorSetLayout,
                                                          1);
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &graphics.descriptorSetPreCompute));

    std::vector<VkWriteDescriptorSet> baseImageWriteDescriptorSets = 
    {
      // Binding 0: Vertex shader uniform buffer
      vks::initializers::writeDescriptorSet(graphics.descriptorSetPreCompute,
                                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                            0,
                                            &uniformBufferVS.descriptor),
      // Binding 1: Fragment shader texture sampler
      vks::initializers::writeDescriptorSet(graphics.descriptorSetPreCompute,
                                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                            1, 
                                            &textureColorMap.descriptor)
    };
    vkUpdateDescriptorSets(device, baseImageWriteDescriptorSets.size(),
                           baseImageWriteDescriptorSets.data(), 0, NULL);
  }

  // Find and create a compute capable device queue
  void getComputeQueue()
  {
    uint32_t queueFamilyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);
    assert(queueFamilyCount >= 1);

    std::vector<VkQueueFamilyProperties> queueFamilyProperties;
    queueFamilyProperties.resize(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, 
                                             queueFamilyProperties.data());

    // First try to find a dedicated compute queue
    bool computeQueueFound = false;
    for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++) {
      if ((queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
          && ((queueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0)) {
        compute.queueFamilyIndex = i;
        computeQueueFound = true;
        break;
      }
    }

    // If there is no dedicated compute queue, just find the first one supporting compute
    if (!computeQueueFound) {
      for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++) {
        if (queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
          compute.queueFamilyIndex = i;
          computeQueueFound = true;
          break;
        }
      }
    }

    // There must be at least one compute queue family
    assert(computeQueueFound);
    vkGetDeviceQueue(device, compute.queueFamilyIndex, 0, &compute.queue);
  }

  void buildComputeCommandBuffer()
  {
    vkQueueWaitIdle(compute.queue);
    
    VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
    VK_CHECK_RESULT(vkBeginCommandBuffer(compute.commandBuffer, &cmdBufInfo));
    vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      compute.pipelines[compute.pipelineIndex]);
    vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout,
                            0, 1, &compute.descriptorSet, 0, 0);
    vkCmdDispatch(compute.commandBuffer, textureComputeTarget.width / 16,
                  textureComputeTarget.height / 16, 1);
    vkEndCommandBuffer(compute.commandBuffer);
  }

  void prepareCompute()
  {
    getComputeQueue();

    // Create compute pipeline
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = 
    {
      // Binding 0: sampled image (read)
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                    VK_SHADER_STAGE_COMPUTE_BIT,
                                                    0),
      // Binding 1: sampled image (write)
      vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                    VK_SHADER_STAGE_COMPUTE_BIT,
                                                    1),
    };

    VkDescriptorSetLayoutCreateInfo descriptorLayout =
      vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(), setLayoutBindings.size());
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout,
                                                nullptr, &compute.descriptorSetLayout));

    VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
      vks::initializers::pipelineLayoutCreateInfo(&compute.descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo,
                                           nullptr, &compute.pipelineLayout));

    VkDescriptorSetAllocateInfo allocInfo = 
      vks::initializers::descriptorSetAllocateInfo(descriptorPool,
                                                   &compute.descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &compute.descriptorSet));

    std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets  = 
    {
      vks::initializers::writeDescriptorSet(compute.descriptorSet,
                                            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                            0,
                                            &textureColorMap.descriptor),
      vks::initializers::writeDescriptorSet(compute.descriptorSet,
                                            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                            1,
                                            &textureComputeTarget.descriptor)
    };
    vkUpdateDescriptorSets(device, computeWriteDescriptorSets.size(),
                           computeWriteDescriptorSets.data(), 0, NULL);

    // Create compute shader pipelines
    VkComputePipelineCreateInfo computePipelineCreateInfo =
      vks::initializers::computePipelineCreateInfo(compute.pipelineLayout, 0);
    shaderNames = { "sharpen", "edgedetect", "emboss" };
    for (auto& shaderName : shaderNames) {
      std::string fileName = getAssetPath() + "shaders/computeshader/" + shaderName + ".comp.spv";
      computePipelineCreateInfo.stage = loadShader(fileName.c_str(), VK_SHADER_STAGE_COMPUTE_BIT);
      VkPipeline pipeline;
      VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, 
                                               &computePipelineCreateInfo, nullptr, &pipeline));
      compute.pipelines.push_back(pipeline);
    }

    // Separate command pool as queue family for compute may be different
    VkCommandPoolCreateInfo cmdPoolInfo = {};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = compute.queueFamilyIndex;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &compute.commandPool));

    VkCommandBufferAllocateInfo cmdBufAllocateInfo =
      vks::initializers::commandBufferAllocateInfo(compute.commandPool,
                                                   VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                                   1);
    VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &compute.commandBuffer));
        
    VkFenceCreateInfo fenceCreateInfo = vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &compute.fence));

    // Build a singel command buffer containing the compute dispatcs commands
    buildComputeCommandBuffer();                                          
  }

  void buildCommandBuffers()
  {
    // Destroy the previous command buffers
    if (!checkCommandBuffers()) {
      destroyCommandBuffers();
      createCommandBuffers();
    }

    VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

    VkClearValue clearValues[2];
    clearValues[0].color = defaultClearColor;
    clearValues[1].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = width;
    renderPassBeginInfo.renderArea.extent.height = height;
    renderPassBeginInfo.clearValueCount = 2;
    renderPassBeginInfo.pClearValues = clearValues;

    for (int32_t i = 0; i < drawCmdBuffers.size(); ++i) {
      // Set the target frame buffer
      renderPassBeginInfo.framebuffer = frameBuffers[i];
      VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

      // Image memory barrier to wait for the compute shader
      VkImageMemoryBarrier imageMemoryBarrier = {};
      imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
      imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
      imageMemoryBarrier.image = textureComputeTarget.image;
      imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
      imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(drawCmdBuffers[i], 
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           VK_FLAGS_NONE,
                           0, nullptr,
                           0, nullptr,
                           1, &imageMemoryBarrier);
      vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

      VkViewport viewport = vks::initializers::viewport((float)width * 0.5f, (float)height, 0.0f, 1.0f);
      vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

      VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
      vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

      VkDeviceSize offsets[1] = { 0 };
      vkCmdBindVertexBuffers(drawCmdBuffers[i], VERTEX_BUFFER_BIND_ID, 1, &vertexBuffer.buffer, offsets);
      vkCmdBindIndexBuffer(drawCmdBuffers[i], indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

      // Left (pre compute)
      vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                              graphics.pipelineLayout, 0, 1, &graphics.descriptorSetPreCompute,
                              0, NULL);
      vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipeline);
      vkCmdDrawIndexed(drawCmdBuffers[i], indexCount, 1, 0, 0, 0);

      // Right (post compute)
      vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                              graphics.pipelineLayout, 0, 1, &graphics.descriptorSetPostCompute,
                              0, NULL);
      vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipeline);
      viewport.x = (float)width / 2.0f;
      vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
      vkCmdDrawIndexed(drawCmdBuffers[i], indexCount, 1, 0, 0, 0);

      vkCmdEndRenderPass(drawCmdBuffers[i]);

      VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }    
  }

  void prepare()
  {
    VkreBase::prepare();
    loadAssets(); 
    generateQuad();
    setupVertexDescriptions();
    prepareUniformBuffers();
    prepareTextureTarget(&textureComputeTarget, textureColorMap.width,
                         textureColorMap.height, VK_FORMAT_R8G8B8A8_UNORM); 
    setupDescriptorSetLayout();
    preparePipelines();
    setupDescriptorPool(); 
    setupDescriptorSet();
    prepareCompute();
    buildCommandBuffers(); 
    prepared = true;
  }
 
  void draw()
  {
    VkreBase::prepareFrame();

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

    VkreBase::submitFrame();

    vkWaitForFences(device, 1, &compute.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &compute.fence);

    VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
    computeSubmitInfo.commandBufferCount = 1;
    computeSubmitInfo.pCommandBuffers = &compute.commandBuffer;
    VK_CHECK_RESULT(vkQueueSubmit(compute.queue, 1, &computeSubmitInfo, compute.fence));
  }

  virtual void render()
  {
    if (!prepared)
      return;

    draw();

  }

  virtual void viewChanged()
  {
    updateUniformBuffers();
  }

  virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay)
  {
    if (overlay->header("Settings")) {
      if (overlay->comboBox("Shader", &compute.pipelineIndex, shaderNames)) {
        buildComputeCommandBuffer();
      }
    }
  }

};

VulkanCNN *vlkcnn;

static void handleEvent(const xcb_generic_event_t *event)
{
  if (vlkcnn != NULL) {
    vlkcnn->handleEvent(event);
  }
}

int main(const int argc, const char *argv[])
{
  for (size_t i = 0; i < argc; i++) VulkanCNN::args.push_back(argv[i]);

  vlkcnn = new VulkanCNN();
  vlkcnn->initVulkan();
  vlkcnn->setupWindow();
  vlkcnn->prepare();
  vlkcnn->renderLoop();
  delete(vlkcnn);

  return 0;
}

