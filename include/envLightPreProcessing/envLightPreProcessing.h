//
// Created by lei on 09.01.23.
//

#pragma once

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>

#include "sophus/se3.hpp"
#include <unordered_map>
#include "preFilter/preFilter.h"


namespace DSONL{

    template <class T>
    struct hash3d
    {
        size_t operator()(const T &key) const
        {
            float mult = 10000.0;
            size_t hash = 137 * std::round(mult * (key.x+10.0)) + 149 * std::round(mult * (key.y+10.0)) + 163 * std::round(mult * (key.z+10.0));
            return hash;
        }
    };

    template <class T>
    struct equalTo
    {
        bool operator()(const T &key1, const T &key2) const
        {
//            cout<<"using hash!!!!!!!!!!!!!!!!!!"<<endl;
//            cout<<"using hash!!!!!!!!!key1.x !!!!!!!!!"<<key1.x <<endl;
            bool res= key1.x == key2.x && key1.y == key2.y && key1.z == key2.z;
            cout<<"using hash!bool res!!"<< res<<endl;
            return res ;
        }
    };

    struct pointEnvlight {
//        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        pointEnvlight(){
            EnvmapSampler.reserve(2);
        }
        pointEnvlight(pointEnvlight const& );
        Sophus::SE3f envMapPose_world; //Sophus::SE3f* envMapPose_camera;
        cv::Point3f pointBase; // i.e. envMapPose_world.translation();
        std::vector<gli::sampler2d<float>> EnvmapSampler; // gli::sampler2d<float>* prefilteredEnvmapSampler; AND  gli::sampler2d<float>* diffuseSampler;


    };

    class envLight{

    public:

        envLight(int argc, char **argv, string envMap_Folder, string controlPointPose_path);
        ~envLight();


        pcl::PointCloud<pcl::PointXYZ>::Ptr ControlpointCloud;

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

        std::unordered_map<cv::Point3f, pointEnvlight, hash3d<cv::Point3f>, equalTo<cv::Point3f>> envLightMap;

        std::vector<gli::sampler2d<float>> brdfSampler;

        std::vector<pointEnvlight> pointEnv_Vec;





    private:






    };








}





