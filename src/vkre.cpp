/*
 * Vulkan 3D Model Renderer
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
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/cimport.h>

#include <vulkan/vulkan.h>
#include "base/VkreBase.h"
#include "base/VulkanTexture.hpp"
#include "base/VulkanBuffer.hpp"
#include "base/VulkanModel.hpp"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false
#define GRID_DIM 7
#define OBJ_DIM 0.05f

struct Material {
  struct PushBlock {
    float roughness;
    float metallic;
    float r, g, b;
  }params;
  std::string name;
  Material() {};
  Material(std::string n, glm::vec3 c, float r, float m)
    : name(n) {
    params.roughness = r;
    params.metallic = m;
    params.r = c.r;
    params.g = c.g;
    params.b = c.b;
  };
};

class VulkanRenderEngine : public VkreBase
{
public:
  bool wireframe = false;

  struct {
    vks::Texture2D colorMap;
  } textures;

  vks::VertexLayout vertexLayout =
    vks::VertexLayout({ vks::VERTEX_COMPONENT_POSITION,
                        vks::VERTEX_COMPONENT_NORMAL,
                        vks::VERTEX_COMPONENT_UV });

  struct Meshes {
    std::vector<vks::Model> objects;
    int32_t objectIndex = 0;
  } models;

  struct {
    vks::Buffer object;
    vks::Buffer params;
  } uniformBuffers;

  struct UBOMatrices {
    glm::mat4 projection;
    glm::mat4 model;
    glm::mat4 view;
    glm::vec3 camPos;
  } uboMatrices;

  struct UBOParams {
    glm::vec4 lights[4];
  } uboParams;

  struct Pipelines{
    VkPipeline solid;
    VkPipeline wireframe = VK_NULL_HANDLE;
  } pipelines;

  VkPipelineLayout pipelineLayout;
  VkDescriptorSet descriptorSet;
  VkDescriptorSetLayout descriptorSetLayout;

  std::vector<Material> materials;
  int32_t materialIndex = 0;
  std::vector<std::string> materialNames;
  std::vector<std::string> objectNames;

  VulkanRenderEngine() : VkreBase(ENABLE_VALIDATION)
  {
    zoom = 1.0f;
    zoomSpeed = 2.5f;
    rotationSpeed = 0.5f;
    rotation = { -0.5f, -112.75f, 0.0f };
    cameraPos = { 1.2f, 0.03f, -1.6f };

    paused = true;
    timerSpeed *= 0.25f;

    title = "VKRE Model rendering";
    settings.overlay = true;

    // Setup materials
    // source:
    // https://seblagarde.wordpress.com/2011/08/17/
    // feeding-a-physical-based-lighting-mode/
    materials.push_back(
      Material("Gold", glm::vec3(1.0f, 0.765557f, 0.336057f), 0.1f, 1.0f));
    materials.push_back(
      Material("Copper", glm::vec3(0.955008f, 0.637427f, 0.538163f), 0.1f, 1.0f));
    materials.push_back(
      Material("Chromium", glm::vec3(0.549585f, 0.556114f, 0.554256f), 0.1f, 1.0f));
    materials.push_back(
      Material("Nickel", glm::vec3(0.659777f, 0.608679f, 0.525649f), 0.1f, 1.0f));
    materials.push_back(
      Material("Titanium", glm::vec3(0.541931f, 0.496791f, 0.449419f), 0.1f, 1.0f));
    materials.push_back(
      Material("Cobalt", glm::vec3(0.662124f, 0.654864f, 0.633732f), 0.1f, 1.0f));
    materials.push_back(
      Material("Platinum", glm::vec3(0.672411f, 0.637331f, 0.585456f), 0.1f, 1.0f));
    materials.push_back(
      Material("White", glm::vec3(1.0f), 0.1f, 1.0f));
    materials.push_back(
      Material("Red", glm::vec3(1.0f, 0.0f, 0.0f), 0.1f, 1.0f));
    materials.push_back(
      Material("Blue", glm::vec3(0.0f, 0.0f, 1.0f), 0.1f, 1.0f));
    materials.push_back(
      Material("Black", glm::vec3(0.0f), 0.1f, 1.0f));

    for (auto material : materials) {
      materialNames.push_back(material.name);
    }

    objectNames = {"Sphere", "Teapot", "Torusknot", "Venus"};
    materialIndex = 9;
  }

  ~VulkanRenderEngine()
  {
    vkDestroyPipeline(device, pipelines.solid, nullptr);
    if (pipelines.wireframe != VK_NULL_HANDLE) {
      vkDestroyPipeline(device, pipelines.wireframe, nullptr);
    }

    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

    for (auto& model : models.objects) {
      model.destroy();
    }

    uniformBuffers.object.destroy();
    uniformBuffers.params.destroy();
  }

  virtual void getEnabledFeatures()
  {
    // Fill mode non solid is required for wireframe display
    if (deviceFeatures.fillModeNonSolid) {
      enabledFeatures.fillModeNonSolid = VK_TRUE;
    };
  }

  void loadAssets()
  {
    std::vector<std::string> filenames = {"f15.dae", "geosphere.obj", "teapot.dae", "torusknot.obj", "venus.fbx"};
    for (auto file : filenames) {
      vks::Model model;
      model.loadFromFile(getAssetPath() + "models/" + file, vertexLayout, OBJ_DIM * (file == "venus.fbx" ? 3.0f : 1.0f), vulkanDevice, queue);
      models.objects.push_back(model);
    }
  }

  void updateLights()
  {
    const float p = 15.0f;
    uboParams.lights[0] = glm::vec4(-p, -p*0.5f, -p, 1.0f);
    uboParams.lights[1] = glm::vec4(-p, -p*0.5f, p, 1.0f);
    uboParams.lights[2] = glm::vec4(p, -p*0.5f, p, 1.0f);
    uboParams.lights[3] = glm::vec4(p, -p*0.5f, -p, 1.0f);

    if (!paused) {
      uboParams.lights[0].x = sin(glm::radians(timer * 360.0f)) * 20.0f;
      uboParams.lights[0].z = cos(glm::radians(timer * 360.0f)) * 20.0f;
      uboParams.lights[1].x = cos(glm::radians(timer * 360.0f)) * 20.0f;
      uboParams.lights[1].y = sin(glm::radians(timer * 360.0f)) * 20.0f;
    }

    memcpy(uniformBuffers.params.mapped, &uboParams, sizeof(uboParams));
  }

  void updateUniformBuffers()
  {
    uboMatrices.projection = glm::perspective(glm::radians(60.0f),
                                        (float)width / (float)height,
                                        0.1f,
                                        256.0f);

    glm::mat4 viewMatrix = glm::translate(glm::mat4(1.0f),
                                          glm::vec3(0.0f, 0.0f, zoom));
    uboMatrices.model = viewMatrix * glm::translate(glm::mat4(1.0f), cameraPos);
    uboMatrices.model = glm::rotate(uboMatrices.model, glm::radians(rotation.x),
                              glm::vec3(1.0f, 0.0f, 0.0f));
    uboMatrices.model = glm::rotate(uboMatrices.model, glm::radians(rotation.y),
                              glm::vec3(0.0f, 1.0f, 0.0f));
    uboMatrices.model = glm::rotate(uboMatrices.model, glm::radians(rotation.z),
                              glm::vec3(0.0f, 0.0f, 1.0f));

    uboMatrices.camPos = cameraPos * -1.0f;

    memcpy(uniformBuffers.object.mapped, &uboMatrices, sizeof(uboMatrices));
  }

  // Prepare and initialize uniform buffer containing shader uniforms
  void prepareUniformBuffers()
  {
    // Object vertex shader  uniform buffer
    VK_CHECK_RESULT(vulkanDevice->createBuffer(
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      &uniformBuffers.object,
      sizeof(uboMatrices)));

    // Shared parameter uniform buffer
    VK_CHECK_RESULT(vulkanDevice->createBuffer(
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      &uniformBuffers.params,
      sizeof(uboParams)));

    // Map persistent
    VK_CHECK_RESULT(uniformBuffers.object.map());
    VK_CHECK_RESULT(uniformBuffers.params.map());

    updateUniformBuffers();
    updateLights();
  }

  void setupDescriptorSetLayout()
  {
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
      // Binding 0: Vertex shader uniform buffer
      vks::initializers::descriptorSetLayoutBinding(
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0),
      // Binding 1: Fragment shader uniform buffer
      vks::initializers::descriptorSetLayoutBinding(
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_SHADER_STAGE_FRAGMENT_BIT,
        1)
    };

    VkDescriptorSetLayoutCreateInfo descriptorLayout =
      vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
      VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device,
                                                 &descriptorLayout,
                                                 nullptr,
                                                 &descriptorSetLayout));

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
    vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout,
                                                1);

    std::vector<VkPushConstantRange> pushConstantRanges = {
      vks::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(glm::vec3), 0),
      vks::initializers::pushConstantRange(VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(Material::PushBlock), sizeof(glm::vec3)),
    };

    pipelineLayoutCreateInfo.pushConstantRangeCount = 2;
    pipelineLayoutCreateInfo.pPushConstantRanges = pushConstantRanges.data();

    VK_CHECK_RESULT(vkCreatePipelineLayout(device,
                                           &pipelineLayoutCreateInfo,
                                           nullptr,
                                           &pipelineLayout));
  }

  void setupDescriptorSet()
  {
    // Descriptor Pool
    // Uses one ubo and one combined image sampler
    std::vector<VkDescriptorPoolSize> poolSizes = {
      vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4),
    };

    VkDescriptorPoolCreateInfo descriptorPoolInfo =
      vks::initializers::descriptorPoolCreateInfo(poolSizes, 2);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo,
                                           nullptr, &descriptorPool));

    // Descriptor Sets
    VkDescriptorSetAllocateInfo allocInfo =
      vks::initializers::descriptorSetAllocateInfo(descriptorPool,
        &descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo,
      &descriptorSet));

    std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
      // Binding 0: Vertex shader uniform buffer
      vks::initializers::writeDescriptorSet(
        descriptorSet,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        0,
        &uniformBuffers.object.descriptor),
      // Binding 1: Fragment shader uniform buffer
      vks::initializers::writeDescriptorSet(
        descriptorSet,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        1,
        &uniformBuffers.params.descriptor)
    };

    vkUpdateDescriptorSets(device,
                           static_cast<uint32_t>(writeDescriptorSets.size()),
                           writeDescriptorSets.data(),
                           0,
                           NULL);
  }

  void preparePipelines()
  {
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
      vks::initializers::pipelineInputAssemblyStateCreateInfo(
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        0,
        VK_FALSE);

    VkPipelineRasterizationStateCreateInfo rasterizationState =
      vks::initializers::pipelineRasterizationStateCreateInfo(
        VK_POLYGON_MODE_FILL,
        VK_CULL_MODE_BACK_BIT,
        VK_FRONT_FACE_COUNTER_CLOCKWISE);

    VkPipelineColorBlendAttachmentState blendAttachmentState =
      vks::initializers::pipelineColorBlendAttachmentState(
        0xf,
        VK_FALSE);

    VkPipelineColorBlendStateCreateInfo colorBlendState =
      vks::initializers::pipelineColorBlendStateCreateInfo(
        1,
        &blendAttachmentState);

    VkPipelineDepthStencilStateCreateInfo depthStencilState =
      vks::initializers::pipelineDepthStencilStateCreateInfo(
        VK_FALSE,
        VK_FALSE,
        VK_COMPARE_OP_LESS_OR_EQUAL);

    VkPipelineViewportStateCreateInfo viewportState =
      vks::initializers::pipelineViewportStateCreateInfo(
        1,
        1);

    VkPipelineMultisampleStateCreateInfo multisampleState =
      vks::initializers::pipelineMultisampleStateCreateInfo(
        VK_SAMPLE_COUNT_1_BIT);

    std::vector<VkDynamicState> dynamicStateEnables = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState =
      vks::initializers::pipelineDynamicStateCreateInfo(
        dynamicStateEnables);

    // Load shaders
    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;
    shaderStages[0] = loadShader(
      getAssetPath() + "shaders/mesh/mesh.vert.spv",
      VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(
      getAssetPath() + "shaders/mesh/mesh.frag.spv",
      VK_SHADER_STAGE_FRAGMENT_BIT);

    VkGraphicsPipelineCreateInfo pipelineCreateInfo =
      vks::initializers::pipelineCreateInfo(
        pipelineLayout,
        renderPass);

    pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
    pipelineCreateInfo.pRasterizationState = &rasterizationState;
    pipelineCreateInfo.pColorBlendState = &colorBlendState;
    pipelineCreateInfo.pMultisampleState = &multisampleState;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pDepthStencilState = &depthStencilState;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCreateInfo.pStages = shaderStages.data();

    // Vertex bindings and attributes
    // Binding description
    std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
      vks::initializers::vertexInputBindingDescription(0, vertexLayout.stride(), VK_VERTEX_INPUT_RATE_VERTEX)
    };

    // Attribute descriptions
    std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
      vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0),
      vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 3),
    };

    VkPipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
    vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
    vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
    vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
    vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

    pipelineCreateInfo.pVertexInputState = &vertexInputState;

    depthStencilState.depthWriteEnable = VK_TRUE;
    depthStencilState.depthTestEnable = VK_TRUE;
    rasterizationState.cullMode = VK_CULL_MODE_FRONT_BIT;

    VK_CHECK_RESULT(vkCreateGraphicsPipelines(
      device,
      pipelineCache,
      1,
      &pipelineCreateInfo,
      nullptr,
      &pipelines.solid));

    // Wire frame rendering pipeline
    if (deviceFeatures.fillModeNonSolid) {
      rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
      rasterizationState.lineWidth = 1.0f;
      VK_CHECK_RESULT(vkCreateGraphicsPipelines(
        device,
        pipelineCache,
        1,
        &pipelineCreateInfo,
        nullptr,
        &pipelines.wireframe));
    }
  }

  void buildCommandBuffers()
  {
    VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

    VkClearValue clearValues[2];
    clearValues[0].color = {{0.35f, 0.35f, 0.5f, 1.0f}} ;
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = width;
    renderPassBeginInfo.renderArea.extent.height = height;
    renderPassBeginInfo.clearValueCount = 2;
    renderPassBeginInfo.pClearValues = clearValues;

    for (int32_t i = 0; i < drawCmdBuffers.size(); ++i) {
      // Set target frame buffer
      renderPassBeginInfo.framebuffer = frameBuffers[i];

      VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

      vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

      VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
      vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

      VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
      vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

      VkDeviceSize offsets[1] = {0};


      vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                        wireframe ? pipelines.wireframe : pipelines.solid);

      vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipelineLayout, 0, 1, &descriptorSet, 0, NULL);

      // Bind mesh vertex buffer
      vkCmdBindVertexBuffers(drawCmdBuffers[i], VERTEX_BUFFER_BIND_ID, 1, &models.objects[models.objectIndex].vertices.buffer, offsets);

      // Bind mesh index buffer
      vkCmdBindIndexBuffer(drawCmdBuffers[i], models.objects[models.objectIndex].indices.buffer, 0, VK_INDEX_TYPE_UINT32);

      Material mat = materials[materialIndex];

#if 1
      mat.params.metallic = 0.5;
      mat.params.roughness = 0.5;
      uint32_t objcount = 1;
      for (uint32_t x = 0; x < objcount; x++) {
        glm::vec3 pos = glm::vec3(float(x - (objcount / 2.0f)) * 2.5f, 0.0f, 0.0f);
        //mat.params.roughness = glm::clamp((float)5 / (float)objcount, 0.005f, 1.0f);
        vkCmdPushConstants(drawCmdBuffers[i], pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::vec3), &pos);
        vkCmdPushConstants(drawCmdBuffers[i], pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(glm::vec3), sizeof(Material::PushBlock), &mat);
        // Render mesh vertex buffer using its indices
        vkCmdDrawIndexed(drawCmdBuffers[i], models.objects[models.objectIndex].indexCount, 1, 0, 0, 0);
      }
#else
      for (uint32_t y = 0; y < GRID_DIM; y++) {
        for (uint32_t x = 0; x < GRID_DIM; x++) {
          glm::vec3 pos = glm::vec3(float(x - (GRID_DIM / 2.0f)) * 2.5f, 0.0f, float(y - (GRID_DIM / 2.0f)) * 2.5f);
          vkCmdPushConstants(drawCmdBuffers[i], pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::vec3), &pos);
          mat.params.metallic = glm::clamp((float)x / (float)(GRID_DIM - 1), 0.1f, 1.0f);
          mat.params.roughness = glm::clamp((float)y / (float)(GRID_DIM - 1), 0.05f, 1.0f);
          vkCmdPushConstants(drawCmdBuffers[i], pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(glm::vec3), sizeof(Material::PushBlock), &mat);
          vkCmdDrawIndexed(drawCmdBuffers[i], models.objects[models.objectIndex].indexCount, 1, 0, 0, 0);
        }
      }
#endif

      vkCmdEndRenderPass(drawCmdBuffers[i]);

      VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }
  }

  void prepare()
  {
    VkreBase::prepare();
    loadAssets();
    prepareUniformBuffers();
    setupDescriptorSetLayout();
    preparePipelines();
    setupDescriptorSet();
    buildCommandBuffers();
    prepared = true;
  }

  void draw()
  {
    VkreBase::prepareFrame();

    // Command buffer to be sumitted to the queue
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];

    // Submit to queue
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

    VkreBase::submitFrame();
  }

  virtual void render()
  {
    if (!prepared)
      return;
    draw();
    if (!paused)
      updateLights();
  }

  virtual void viewChanged()
  {
    updateUniformBuffers();
  }

  virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay)
  {
    if (overlay->header("Settings")) {
      if (overlay->checkBox("Wireframe", &wireframe)) {
        buildCommandBuffers();
      }
    }
  }
};

VulkanRenderEngine *vkre;

static void handleEvent(const xcb_generic_event_t *event)
{
  if (vkre != NULL) {
    vkre->handleEvent(event);
  }
}

int main(const int argc, const char *argv[])
{
  for (size_t i = 0; i < argc; i++) VulkanRenderEngine::args.push_back(argv[i]);

  vkre = new VulkanRenderEngine();
  vkre->initVulkan();
  vkre->setupWindow();
  vkre->prepare();
  vkre->renderLoop();
  delete(vkre);

  return 0;
}

