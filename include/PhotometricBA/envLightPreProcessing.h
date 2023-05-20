//
// Created by lei on 05.05.23.
//

#ifndef NLDSO_PHOTOMETRICLOSS_ENVLIGHTPREPROCESSING_H
#define NLDSO_PHOTOMETRICLOSS_ENVLIGHTPREPROCESSING_H

#pragma once

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include "sophus/se3.hpp"
#include <unordered_map>

#include "preFilter/preFilter.h"
#include "settings/common.h"



namespace PBANL{

    void readCtrlPointPoseData(string fileName, vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>>& pose, Sophus::SE3f frontCamPose);



    struct pointEnvlight {
//        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        pointEnvlight(){
            EnvmapSampler.reserve(2);
        }
        pointEnvlight(pointEnvlight const& );
        Sophus::SE3f envMapPose_world; //Sophus::SE3f* envMapPose_camera;
        cv::Point3f pointBase; // i.e. envMapPose_world.translation();
        std::vector<gli::sampler2d<float>> EnvmapSampler; // gli::sampler2d<float>* prefilteredEnvmapSampler; AND  gli::sampler2d<float>* diffuseSampler;
        int ctrlPointIdx;

    };


    class envLightLookup{

    public:

        envLightLookup(int argc, char **argv, string envMap_Folder, string controlPointPose_path, Sophus::SE3f frontCamPose);
        ~envLightLookup();

        pcl::PointCloud<pcl::PointXYZ>::Ptr ControlpointCloud;

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

        std::unordered_map<cv::Point3f, int, DSONL::hash3d<cv::Point3f>, DSONL::equalTo<cv::Point3f>> envLightIdxMap;

        std::vector<gli::sampler2d<float>> brdfSampler;

    };









}



#endif //NLDSO_PHOTOMETRICLOSS_ENVLIGHTPREPROCESSING_H
