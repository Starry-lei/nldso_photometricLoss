//
// Created by cheng on 27.11.22.
//

#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "sophus/se3.hpp"
//#include <opencv2/core/mat.h>
//#include <opencv2/imgproc.h>
#include <GL/glew.h>
//#include <opencv2/core/core.h>
#include <string>
#include <iostream>
#include <thread>
#include "wrapperHeader.h"

//#include "gli/gli/sampler2d.hpp"

using namespace cv;

namespace DSONL{





	extern std::vector<cv::Mat> img_pyramid;
    extern std::vector<cv::Mat> img_pyramid_mask;

	extern  cv::Mat img_diffuseMap;
	extern  cv::Mat img_diffuseMapMask;

    extern  const  gli::sampler2d<float>* prefilteredEnvmapSampler;
    extern  const  gli::sampler2d<float>* diffuseSampler;
    extern  const  gli::sampler2d<float>* brdfSampler_;

    // extern Vec2i boundingBoxUpperLeft(83, 76);
    // extern Vec2i boundingBoxBotRight(240, 320);





}

