cmake_minimum_required(VERSION 3.10)

# set the project name
project(shader)
# add the executable
add_executable(shader main.cpp FileTools.cpp FileTools.h LoadOBJ.cpp LoadOBJ.h Matrix.cpp Matrix.h Mesh.cpp Mesh.h Renderer.cpp Renderer.h ShaderNode.cpp ShaderNode.h StringTools.cpp StringTools.h)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})

target_link_libraries(shader ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${GLEW_LIBRARIES})
