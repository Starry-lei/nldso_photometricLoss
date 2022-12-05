//
// Created by cheng on 05.12.22.
//

#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include <math.h> /* fmod */
#include "iostream"
using namespace cv;
namespace DSONL
{

        // added by Binghui
        float dot(const Eigen::Vector3f, const Eigen::Vector3f);

        float mod(const float, const float);

        float clamp(const float, const float, const float);

        float pow(const float, const float);
        Eigen::Vector3f pow(const float, const Eigen::Vector3f);
        //        template typename  TEigen::Matrix<T, 3,1>;
        //        Vec3f
        Eigen::Vector3f pow(const Eigen::Vector3f, const float);
        Eigen::Vector3f pow(const Eigen::Vector3f, const Eigen::Vector3f);

        Eigen::Vector3f normalize(const Eigen::Vector3f);

        float mix(const float, const float, const float);
        Eigen::Vector3f mix(const Eigen::Vector3f, const Eigen::Vector3f, const float);

        Eigen::Vector3f reflect(const Eigen::Vector3f, const Eigen::Vector3f);

        class IBL_Radiance
        {

        public:
                IBL_Radiance();
                ~IBL_Radiance();
                Vec2f directionToSphericalEnvmap(Vec3f dir);
                Vec3f specularIBL(Vec3f F0, float roughness, Vec3f N, Vec3f V);
                Vec3f diffuseIBL(Vec3f normal);
                Vec3f fresnelSchlick(float cosTheta, Vec3f F0);
                Vec3f ibl_radiance_val;

        private:
        };

}
