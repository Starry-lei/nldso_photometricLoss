//
// Created by cheng on 27.11.22.
//
#include "preComputeSetting.h"

namespace DSONL{
      std::vector<cv::Mat> img_pyramid;
      cv::Mat img_diffuseMap;
      cv::Mat img_diffuseMapMask;
      gli::sampler2d<float>* prefilteredEnvmapSampler =NULL;
      gli::sampler2d<float>* brdfSampler= NULL;
}