cmake_minimum_required(VERSION 3.10)

project(nldso_photometricLoss)

# FindEigen3 (for opengv / Pangolin) and FindTBB
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" ${CMAKE_MODULE_PATH})

# find eigen with our adapted "config module", that uses our eigen source tree
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/eigen3" ${CMAKE_PREFIX_PATH})

set(EIGEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/eigen")
set(CERES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/build-ceres-solver")

 set(PANGOLIN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/build-Pangolin/src")



option(BUILD_STATIC   "Build static libraries"  ON)
option(WITH_TBB      "with Intel TBB"          ON)
option(WITH_OPENMP   "with openmp"             ON)
option(WITH_OPENCV   "with opencv"             ON)
option(WITH_MATLAB   "compile with Matlab interface" OFF)



# set(OPENGV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/build-opengv")
# set(g2o_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/g2o/lib/cmake/g2o")

# Set default build type if not specified otherwise.
# See https://cmake.org/pipermail/cmake/2012-May/050243.html
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified.")

  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# Configure CCache if available
if(NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
  find_program(CCACHE_PROGRAM ccache)

  if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  else()
    message(STATUS "Dind't find ccache")
  endif()
else()
  message(STATUS "Compiler launcher already set. Not configuring ccache.")
  message(STATUS "CMAKE_C_COMPILER_LAUNCHER: ${CMAKE_C_COMPILER_LAUNCHER}")
  message(STATUS "CMAKE_CXX_COMPILER_LAUNCHER: ${CMAKE_CXX_COMPILER_LAUNCHER}")
endif()

# See comment in ./build_submodules.sh on the relation of arch=native
# and Eigen's alignment. TLDR: Passing arch=native for all build types
# is currently the only viable option to avoid subtle bugs with Eigen.
SET(CXX_MARCH native)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(BUILD_STATIC)
  set(LIBRARY_TYPE STATIC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
else()
  set(LIBRARY_TYPE SHARED)
endif()






# for some reason clang doesn't include them in Wall whereas gcc does, so include them explicitly.
SET(EXTRA_WARNING_FLAGS "-Wsign-compare")

# clang-specific compile flags to avoid warnings from thirdpary includes
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set(EXTRA_WARNING_FLAGS "${EXTRA_WARNING_FLAGS} -Wno-exceptions")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN") # cmake default: "-g"
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -DNDEBUG -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN") # cmake default: "-O2 -g -DNDEBUG"
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG") # cmake default: "-O3 -DNDEBUG"
SET(CMAKE_CXX_FLAGS " -ftemplate-backtrace-limit=0 -Wall -Wextra ${EXTRA_WARNING_FLAGS} -march=${CXX_MARCH} ${CMAKE_CXX_FLAGS}")

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(EIGEN3_INCLUDE_DIR_HINTS ${EIGEN_ROOT}) # for opengv / pangolin
find_package(Eigen3 3.3.8 EXACT REQUIRED CONFIG) # find eigen with our adapted "config module", that uses our eigen source tree
set(EIGEN_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR}) # for opengv / pangolin
include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS "Found Eigen headers in: ${EIGEN3_INCLUDE_DIR}")

include_directories(thirdparty/Sophus)
include_directories(thirdparty/cereal/include)
include_directories(thirdparty/CLI11/include)

# include_directories(thirdparty/g2o/include)
include_directories(include)

# to use link to target "ceres"
# Note: always link to ceres first, as otherwise a system-wide install might
# be found before our custom build. For example on macOS a both pangolin and
# TBB have /usr/local/include as part of their include directories.
find_package(Ceres 2.0 REQUIRED CONFIG HINTS "${CERES_ROOT}" NO_DEFAULT_PATH)

# to use link to target "pangolin"
set(Pangolin_DIR ${PANGOLIN_ROOT})

 find_package(Pangolin REQUIRED CONFIG NO_DEFAULT_PATH)
 message(STATUS "Found Pangolin headers in: ${Pangolin_INCLUDE_DIR}")

find_package(fmt REQUIRED)
# to use link to target "opengv"
# find_package(opengv REQUIRED CONFIG HINTS ${OPENGV_ROOT} NO_DEFAULT_PATH)
# message(STATUS "Found OpenGV headers in: ${opengv_DIR}")

# to use link to target "TBB"
find_package(TBB REQUIRED)
message(STATUS "Found TBB ${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR} (interface version ${TBB_INTERFACE_VERSION}) headers in: ${TBB_INCLUDE_DIRS}")
list(APPEND MY_LIBRARIES ${TBB_LIBRARIES})
add_definitions(-DWITH_TBB)
# to use link to target "OpenCV"
# NOTE: not specifying version, since 2, 3 or 4 is fine
find_package(OpenCV  REQUIRED)

# COMPONENTS core features2d
message(STATUS "Found OpenCV ${OpenCV_VERSION} headers in: ${OpenCV_INCLUDE_DIRS}")
add_library(OpenCV INTERFACE IMPORTED)
set_property(TARGET OpenCV PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OpenCV_INCLUDE_DIRS})
set_property(TARGET OpenCV PROPERTY INTERFACE_LINK_LIBRARIES ${OpenCV_LIBS})

if(OpenCV_VERSION VERSION_LESS 3)
  set_property(TARGET OpenCV PROPERTY INTERFACE_COMPILE_DEFINITIONS DBATK_OPENCV2)
endif()

# add PCL library, remained to be installed as third party library
if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
  set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

# add PCL library
find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  add_definitions(-DWITH_OPENMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# message(STATUS "show PCL-----${OPENGL_INCLUDE_DIRS}")
# message(STATUS "show PCL-----${GLEW_INCLUDE_DIRS}")
# message(STATUS "show PCL-----PCL_DEFINITIONS: ${PCL_DEFINITIONS}")
# some output:
#message(STATUS "show IT: ${CMAKE_CURRENT_SOURCE_DIR}")

set(dsonl_SOURCE_FILES

  include/settings/setting.cpp
  include/dataLoader/PFMReadWrite.cpp
  include/preFilter/preFilter.h
  # include/numType.h
  include/preFilter/FileTools.cpp
  include/preFilter/Renderer.cpp

  include/preFilter/Matrix.cpp
  include/preFilter/Mesh.cpp
  include/preFilter/ShaderNode.cpp
  include/preFilter/StringTools.cpp
  include/preFilter/LoadOBJ.cpp

  include/preFilter/specular_mask_Renderer.cpp
  include/diffuseMap/Renderer_diffuse.cpp
  include/diffuseMap/diffuse_mask_Renderer.cpp

  include/deltaCompute/deltaCompute.h
  include/deltaCompute/deltaCompute.cpp

  include/envLightPreProcessing/envLightPreProcessing.h
#  include/envLightPreProcessing/envLightPreProcessing.cpp



  include/settings/preComputeSetting.h
  include/settings/preComputeSetting.cpp

  include/utils/ultils.h
  include/cameraProjection/reprojection.h
  include/cameraProjection/photometricBA.h
  include/dataLoader/dataLoader.h

#  include/wrapperHeader.h

        include/envLightPreProcessing/envLightPreProcessing.cpp
        include/preFilter/preFilterUtils.h
        include/preFilter/preFilter.cpp
        include/controlPointSelector/ctrlPointSelector.cpp
        include/controlPointSelector/ctrlPointSelector.h

        include/pixelSelector.h
        include/pixelSelector.cpp

        include/Statistics/Statistics.h
        include/Statistics/IDistribution.h
        include/Statistics/IDistribution.cpp
        include/Statistics/RobustNormalDistribution.h
        include/Statistics/RobustNormalDistribution.cpp

        include/Statistics/TDistribution.h
        include/Statistics/TDistribution.cpp

        include/Utils/Settings.h
        include/Utils/Settings.cpp
        include/Utils/UtilFunctions.h
        include/Visualizer/IVisualizer.h
        include/Visualizer/IVisualizer.cpp



        )


find_package(Boost 1.71.0 REQUIRED COMPONENTS system filesystem program_options)
include_directories( ${Boost_INCLUDE_DIRS})
link_directories( ${Boost_LIBRARY_DIRS})
#add_definitions(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})
#add_definitions(-DWITH_BOOST)
#add_definitions(-DWITH_SIMD)
include_directories(${Boost_INCLUDE_DIRS})
list(APPEND MY_LIBRARIES ${Boost_LIBRARIES})

message(STATUS "show ${Boost_LIBRARIES} Boost_LIBRARIES: ${Boost_INCLUDE_DIRS}")

add_library(dsonl ${dsonl_SOURCE_FILES})
add_subdirectory(include/matplotplusplus)

add_executable(photometricLoss src/photometricLoss.cpp src/SpecularHighlightRemoval/SpecularHighlightRemoval.cpp)
target_link_libraries(photometricLoss
  PUBLIC matplot
  Ceres::ceres
  pangolin
  fmt::fmt
  TBB::tbb
  OpenCV
  ${PCL_LIBRARIES}
  dsonl
  ${OPENGL_LIBRARIES}
  ${GLUT_LIBRARIES}
  ${GLEW_LIBRARIES}
)


add_subdirectory(include/PhotometricBA)

add_executable(pba src/photoBA.cpp include/PhotometricBA/pba.cpp include/PhotometricBA/utils.cpp include/PhotometricBA/trajectory.cpp include/PhotometricBA/pose_utils.cpp
        include/PhotometricBA/imgproc.cpp include/PhotometricBA/dataset.cpp  include/PhotometricBA/file_loader.cpp
        include/pixelSelector.cpp
        include/settings/setting.cpp
        include/Statistics/TDistribution.cpp
        include/Statistics/IDistribution.cpp
        include/Statistics/RobustNormalDistribution.cpp
        include/Statistics/Statistics.h
        include/Statistics/BetaFunction.cpp
        include/Statistics/NelderMead.cpp
        include/Utils/Settings.cpp

        )
target_link_libraries(pba
        Eigen3::Eigen
        ${Boost_LIBRARIES}
        Ceres::ceres
        TBB::tbb
        OpenCV
        pangolin
)

add_subdirectory(QtVisualizer)

#add_executable(test_functions src/test_functions.cpp)
#target_link_libraries(test_functions
#  Ceres::ceres
#  TBB::tbb
#  OpenCV
#  ${PCL_LIBRARIES}
#  dsonl
#  ${OPENGL_LIBRARIES}
#  ${GLUT_LIBRARIES}
#  ${GLEW_LIBRARIES}
#)