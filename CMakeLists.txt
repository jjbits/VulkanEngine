cmake_minimum_required (VERSION 3.8)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake") 

project (VKRE)

find_package(Vulkan)

# get rid of annoying MSVC warnings.
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
add_definitions(-DVK_EXAMPLE_DATA_DIR=\"${CMAKE_SOURCE_DIR}/data/\") 

set (CMAKE_CXX_STANDARD 11)

include_directories(${Vulkan_INCLUDE_DIR})
include_directories(external/imgui) 
include_directories(external/glm)
include_directories(external/gli)
include_directories(external/stb)

find_package(ASSIMP REQUIRED)
find_package(XCB REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_XCB_KHR")


set(ALL_LIBS  ${Vulkan_LIBRARY} ${XCB_LIBRARIES} ${ASSIMP_LIBRARIES})

add_executable(cnn src/base/VkreBase.cpp src/base/VulkanTools.cpp src/base/VulkanDebug.cpp 
                    src/base/VulkanUIOverlay.cpp external/imgui/imgui.cpp external/imgui/imgui_draw.cpp
                    src/cnn.cpp)

set_target_properties(cnn PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

target_link_libraries(cnn ${ALL_LIBS} )

add_executable(vkre src/base/VkreBase.cpp src/base/VulkanTools.cpp src/base/VulkanDebug.cpp 
                    src/base/VulkanUIOverlay.cpp external/imgui/imgui.cpp external/imgui/imgui_draw.cpp
                    src/vkre.cpp)

set_target_properties(vkre PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

target_link_libraries(vkre ${ALL_LIBS} )
