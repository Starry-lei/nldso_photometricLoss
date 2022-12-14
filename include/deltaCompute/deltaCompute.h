//
// Created by cheng on 05.12.22.
//

#pragma once
#include "iostream"
#include "settings/preComputeSetting.h"
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <cmath>
#include <math.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <unordered_map>

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

        Vec3f reflect(Vec3f, Vec3f);



        class IBL_Radiance
        {

        public:
                IBL_Radiance();
                ~IBL_Radiance();
                int mipCount=5;
                Vec3f prefilteredColor(float u, float v, float level);
                Vec2f brdfIntegration(float NoV,float roughness );
                Vec2f directionToSphericalEnvmap(Vec3f dir);
                Vec3f specularIBL(Vec3f F0, float roughness, Vec3f N, Vec3f V, const Eigen::Matrix3d Camera1_c2w);
                Vec3f specularIBLCheck(Vec3f F0, float roughness, Vec3f N, Vec3f V, const Eigen::Matrix3d Camera1_c2w);

		        Vec3f ACESFilm(Vec3f radiance);

                Vec3f diffuseIBL(Vec3f normal);
                Vec3f fresnelSchlick(float cosTheta, Vec3f F0);
                Vec3f ibl_radiance_val;
                Vec3f solveForRadiance(Vec3f viewDir, Vec3f normal,  const float& roughnessValue,
                                       const float& metallicValue,
                                       const float &reflectance,
                                       const Vec3f& baseColorValue,
                                       const Eigen::Matrix3d Transformation_wc
                                       );




        private:
        };

        void updateDelta(
            const Eigen::Matrix3d Camera1_c2w,
			// const Sophus::SE3d& CurrentT,
            Sophus::SO3d& Rotation,
            Eigen::Matrix<double, 3, 1>& Translation,
            const Eigen::Matrix3f& K,
            const Mat& image_baseColor,
            const Mat depth_map,
            const float& image_metallic,
            const float& image_roughnes,
            Mat& deltaMap,
            Mat& newNormalMap,
            float& upper_b,
            float& lower_b
        );





}
