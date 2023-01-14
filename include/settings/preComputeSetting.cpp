//
// Created by cheng on 27.11.22.
//
#include "preComputeSetting.h"

namespace DSONL{
      std::vector<cv::Mat> img_pyramid;
      std::vector<cv::Mat> img_pyramid_mask;

      cv::Mat img_diffuseMap;
      cv::Mat img_diffuseMapMask;
    const gli::sampler2d<float>* prefilteredEnvmapSampler =NULL;
    const gli::sampler2d<float>* brdfSampler_=NULL;
    const  gli::sampler2d<float>* diffuseSampler= NULL;



}