file(GLOB PhotometricBA "*.cpp")
add_library(photobundle ${LIBRARY_TYPE} ${PhotometricBA} pangolinVis.cpp pangolinVis.h envLightPreProcessing.cpp envLightPreProcessing.h)

set_target_properties(photobundle PROPERTIES LINKER_LANGUAGE CXX)
option(WITH_OPENCV   "with opencv"             ON)

target_link_libraries(photobundle ${OpenCV_LIBS} Eigen3::Eigen ${MY_LIBRARIES} ${CERES_LIBRARIES})
