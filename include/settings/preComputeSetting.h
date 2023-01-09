//
// Created by cheng on 27.11.22.
//

#pragma once

#include "sophus/se3.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <GL/glew.h>
#include <opencv2/core/core.hpp>
#include <string>
#include <iostream>
#include <thread>
#include "wrapperHeader.h"

//#include "gli/gli/sampler2d.hpp"

using namespace cv;

namespace DSONL{


     struct envMap {
            Sophus::SE3f envMapPose_world; //Sophus::SE3f* envMapPose_camera;
            cv::Point3f pointBase; // i.e. envMapPose_world.translation();
            gli::sampler2d<float>* prefilteredEnvmapSampler;
            gli::sampler2d<float>* diffuseSampler;
    };






	extern std::vector<cv::Mat> img_pyramid;
    extern std::vector<cv::Mat> img_pyramid_mask;

	extern  cv::Mat img_diffuseMap;
	extern  cv::Mat img_diffuseMapMask;

    extern  gli::sampler2d<float>* prefilteredEnvmapSampler;
    extern  gli::sampler2d<float>* diffuseSampler;
    extern  gli::sampler2d<float>* brdfSampler;



}

