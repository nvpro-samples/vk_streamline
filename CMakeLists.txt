cmake_minimum_required(VERSION 3.5)

get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME})

message(STATUS "-------------------------------")
message(STATUS "Processing Project ${PROJECT_NAME}:")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 20)

set(STREAMLINE_MANUAL_HOOKING ON CACHE BOOL "Use manual hooking method of Streamline instead of linking against the interposer")

#--------------------------------------------------------------------------------------------------
# look for nvpro_core 1) as a sub-folder 2) at some other locations
# this cannot be put anywhere else since we still didn't find setup.cmake yet
#
if(NOT BASE_DIRECTORY)
  find_path(BASE_DIRECTORY
    NAMES nvpro_core/cmake/setup.cmake
    PATHS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../.. 
    REQUIRED
    DOC "Directory containing nvpro_core"
    )
endif()

if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/utilities.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
endif()

set(NVPRO_CORE_DIR ${BASE_DIRECTORY}/nvpro_core)
set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(STREAMLINE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/streamline CACHE PATH "Path to the Streamline SDK.")

macro(_add_package_Streamline)
  if(STREAMLINE_ROOT_DIR AND EXISTS ${STREAMLINE_ROOT_DIR})
    set(STREAMLINE_INCLUDE_DIR ${STREAMLINE_ROOT_DIR}/include)
    set(STREAMLINE_LIBRARY ${STREAMLINE_ROOT_DIR}/lib/x64/sl.interposer.lib)

    set(STREAMLINE_DLLS
        ${STREAMLINE_ROOT_DIR}/bin/x64/sl.interposer.dll
        ${STREAMLINE_ROOT_DIR}/bin/x64/sl.common.dll
        ${STREAMLINE_ROOT_DIR}/bin/x64/sl.dlss.dll
        ${STREAMLINE_ROOT_DIR}/bin/x64/sl.dlss_g.dll
        ${STREAMLINE_ROOT_DIR}/bin/x64/sl.reflex.dll
        ${STREAMLINE_ROOT_DIR}/bin/x64/NvLowLatencyVk.dll
        ${STREAMLINE_ROOT_DIR}/bin/x64/nvngx_dlss.dll
        ${STREAMLINE_ROOT_DIR}/bin/x64/nvngx_dlssg.dll
        )

    include_directories(${STREAMLINE_INCLUDE_DIR})

    if(STREAMLINE_MANUAL_HOOKING)
      add_definitions(-DSL_MANUAL_HOOKING=1)
    else()
      add_definitions(-DSL_MANUAL_HOOKING=0)

      # Replace Vulkan library with Streamline interposer
      list(APPEND LIBRARIES_OPTIMIZED ${STREAMLINE_LIBRARY})
      list(APPEND LIBRARIES_DEBUG ${STREAMLINE_LIBRARY})
      list(REMOVE_ITEM LIBRARIES_OPTIMIZED ${Vulkan_LIBRARY})
      list(REMOVE_ITEM LIBRARIES_DEBUG ${Vulkan_LIBRARY})
      list(REMOVE_ITEM PLATFORM_LIBRARIES ${Vulkan_LIBRARY})

      set(Vulkan_LIBRARY ${STREAMLINE_LIBRARY})
      set(Vulkan_LIBRARIES ${Vulkan_LIBRARY})
    endif()
  else()
    message("Streamline SDK not found.")
  endif()
endmacro()

_add_package_VulkanSDK()
_add_package_VMA()
_add_package_ImGUI()
_add_package_Streamline() # Add after Vulkan SDK package, to potentially replace Vulkan library

_add_nvpro_core_lib()

#####################################################################################
# Source files for this project
#
file(GLOB SOURCE_FILES
    *.cpp
    *.hpp
    *.h
    )

file(GLOB SHADER_SOURCE_FILES  # Shader sources
    ${SHADER_DIR}/*.vert
    ${SHADER_DIR}/*.frag
    ${SHADER_DIR}/*.rgen
    ${SHADER_DIR}/*.rchit
    ${SHADER_DIR}/*.rahit
    ${SHADER_DIR}/*.rmiss
    ${SHADER_DIR}/*.comp
    )
    
file(GLOB SHADER_HEADER_FILES  # Shader headers
    ${SHADER_DIR}/*.glsl 
    ${SHADER_DIR}/*.h
    )

source_group("Shader Files" FILES ${SHADER_SOURCE_FILES} ${SHADER_HEADER_FILES})

#####################################################################################
# Executable
#
add_executable(${PROJECT_NAME}
    ${SOURCE_FILES}
    ${COMMON_SOURCE_FILES}
    ${PACKAGE_SOURCE_FILES}
    )

target_sources(${PROJECT_NAME} PRIVATE
    ${SHADER_SOURCE_FILES}
    ${SHADER_HEADER_FILES}
    )

target_link_libraries(${PROJECT_NAME} 
    ${PLATFORM_LIBRARIES} 
    ${LIBRARIES_OPTIMIZED} 
    nvpro_core
    )

_add_project_definitions(${PROJECT_NAME})

compile_glsl(
  SOURCE_FILES ${SHADER_SOURCE_FILES}
  HEADER_FILES ${SHADER_HEADER_FILES}
  DST "${CMAKE_CURRENT_SOURCE_DIR}/_autogen" 
  VULKAN_TARGET "vulkan1.2"
  HEADER ON
  DEPENDENCY ${VULKAN_BUILD_DEPENDENCIES}
  FLAGS -I${SHADER_DIR} -I${NVPRO_CORE_DIR} -g
  )

_finalize_target(${PROJECT_NAME})
_copy_files_to_target(${PROJECT_NAME} "${STREAMLINE_DLLS}")
install(FILES ${STREAMLINE_DLLS} CONFIGURATIONS Release DESTINATION bin_${ARCH})
install(FILES ${STREAMLINE_DLLS} CONFIGURATIONS Debug DESTINATION bin_${ARCH}_debug)
