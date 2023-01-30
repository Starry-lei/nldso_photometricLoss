//
// Created by lei on 09.01.23.
//

#pragma once

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include "sophus/se3.hpp"
#include <unordered_map>

#include "preFilter/preFilter.h"
#include "settings/common.h"



namespace DSONL{

    void readCtrlPointPoseData(string fileName, vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>>& pose);



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

    class envLight{

    public:

        envLight(std::unordered_map<int, int> selectedIndex, int argc, char **argv, string envMap_Folder, string controlPointPose_path);
        ~envLight();


        pcl::PointCloud<pcl::PointXYZ>::Ptr ControlpointCloud;

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

        std::unordered_map<cv::Point3f, pointEnvlight, hash3d<cv::Point3f>, equalTo<cv::Point3f>> envLightMap;

        std::vector<gli::sampler2d<float>> brdfSampler;

        std::vector<pointEnvlight> pointEnv_Vec;

    private:
    };



    class envLightLookup{

    public:

        envLightLookup(std::unordered_map<int, int> selectedIndex, int argc, char **argv, string envMap_Folder, string controlPointPose_path);
        ~envLightLookup();

        pcl::PointCloud<pcl::PointXYZ>::Ptr ControlpointCloud;

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

        std::unordered_map<cv::Point3f, int, hash3d<cv::Point3f>, equalTo<cv::Point3f>> envLightIdxMap;

        std::vector<gli::sampler2d<float>> brdfSampler;

    };









}





