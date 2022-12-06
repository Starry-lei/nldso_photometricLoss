//
// Created by cheng on 05.12.22.
//

#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include <math.h> /* fmod */
#include <cmath>

#include "iostream"
using namespace cv;
namespace DSONL
{


        // added by Binghui
	double dot(const Eigen::Vector3d, const Eigen::Vector3d);

	double mod(const double, const double);

	double clamp(const double, const double, const double);

	double pow(const double, const double);
	Eigen::Vector3d pow(const double, const Eigen::Vector3d);
//        template typename  TEigen::Matrix<T, 3,1>;
//        Vec3f
	Eigen::Vector3d pow(const Eigen::Vector3d, const double);
	Eigen::Vector3d pow(const Eigen::Vector3d, const Eigen::Vector3d);

	Eigen::Vector3d normalize(const Eigen::Vector3d);

	double mix(const double, const double, const double);

	Eigen::Vector3d mix(const Eigen::Vector3d, const Eigen::Vector3d, const double);

	Eigen::Vector3d reflect(const Eigen::Vector3d, const Eigen::Vector3d);


        class IBL_Radiance{

        public:
          IBL_Radiance();
          ~IBL_Radiance();
          Vec2f directionToSphericalEnvmap(Vec3f dir);
          Vec3f specularIBL(Vec3f F0 , float roughness, Vec3f N, Vec3f V);
          Vec3f diffuseIBL(Vec3f normal);
          Vec3f fresnelSchlick(float cosTheta, Vec3f F0);
          Vec3f ibl_radiance_val;


        private:

        };















}
