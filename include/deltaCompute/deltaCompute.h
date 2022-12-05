//
// Created by cheng on 05.12.22.
//

#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Core>
#include <math.h> /* fmod */
#include "iostream"

namespace DSONL
{

	//TODO:
	//  1. mod
	//  2. reflect
	//  3. clamp
	//  4. normalize
	//  5. pow
	//  6. mix
	//  7. dot

	double dot(const Eigen::Vector3d, const Eigen::Vector3d);

	double mod(const double, const double);

	double clamp(const double, const double, const double);

	double pow(const double, const double);
	Eigen::Vector3d pow(const double, const Eigen::Vector3d);
	Eigen::Vector3d pow(const Eigen::Vector3d, const double);
	Eigen::Vector3d pow(const Eigen::Vector3d, const Eigen::Vector3d);

	Eigen::Vector3d normalize(const Eigen::Vector3d);

	double mix(const double, const double, const double);
	Eigen::Vector3d mix(const Eigen::Vector3d, const Eigen::Vector3d, const double);

	Eigen::Vector3d reflect(const Eigen::Vector3d, const Eigen::Vector3d);
}
