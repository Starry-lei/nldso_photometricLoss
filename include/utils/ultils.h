//
// Created by cheng on 13.09.22.
//
#pragma once


#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include "sophus/se3.hpp"

#include <iostream>
#include <vector>
#include <cmath>


//
//#include <ceres/ceres.h>
//#include <ceres/cubic_interpolation.h>
//#include <ceres/loss_function.h>

//#include <omp.h>
//#include <pcl/io/ply_io.h>
//#include <pcl/io/pcd_io.h>
//#include <chrono>
//#include <pcl/point_cloud.h>
//#include <pcl/kdtree/kdtree_flann.h>
//#include <pcl/search/search.h>
//#include <pcl/surface/mls.h>
//
//#include <pcl/point_types.h>
//#include <pcl/io/io.h>
//
//#include <pcl/visualization/cloud_viewer.h>
//#include <pcl/filters/radius_outlier_removal.h>
//#include <pcl/filters/voxel_grid.h>
//#include <pcl/filters/statistical_outlier_removal.h>
//#include <pcl/features/normal_3d.h>


namespace DSONL {

	using namespace cv;
	using namespace std;
	const double DEG_TO_ARC = 0.0174532925199433;


	bool AddGaussianNoise_Opencv(const Mat mSrc, Mat &mDst, double Mean = 0.0, double StdDev = 10.0) {
		if (mSrc.empty()) {
			cout << "[Error]! Input Image Empty!";
			return 0;
		}

		Mat mSrc_32FC1;
		Mat mGaussian_noise = Mat(mSrc.size(), CV_64FC1);
		randn(mGaussian_noise, Scalar::all(Mean), Scalar::all(StdDev));

		mSrc.convertTo(mSrc_32FC1, CV_64FC1);
		addWeighted(mSrc_32FC1, 1.0, mGaussian_noise, 1.0, 0.0, mSrc_32FC1);
		mSrc_32FC1.convertTo(mDst, mSrc.type());

		return true;
	}

	bool AddGaussianNoise_Opencv(const Mat mSrc, Mat &mDst, double Mean, double StdDev, float *statusMap) {
		if (mSrc.empty()) {
			cout << "[Error]! Input Image Empty!";
			return 0;
		}

		Mat mSrc_32FC1;
		Mat mGaussian_noise = Mat(mSrc.size(), CV_64FC1);
		randn(mGaussian_noise, Scalar::all(Mean), Scalar::all(StdDev));
		mSrc.convertTo(mSrc_32FC1, CV_64FC1);
		for (int u = 0; u < mSrc.rows; u++) // colId, cols: 0 to 480
		{
			for (int v = 0; v < mSrc.cols; v++) // rowId,  rows: 0 to 640
			{
				if (statusMap != NULL && statusMap[u * mSrc.cols + v] != 0) {
					mSrc_32FC1.at<double>(u, v) += mGaussian_noise.at<double>(u, v);

				}
			}
		}
//		addWeighted(mSrc_32FC1, 1.0, mGaussian_noise, 1.0, 0.0, mSrc_32FC1);
		mSrc_32FC1.convertTo(mDst, mSrc.type());

		return true;
	}


	// show image function
	void imageInfo(Mat im_, Eigen::Vector2i &position) {

		//position,  fst is rowIdx, snd is colIdx
		Mat im = im_;
		int img_depth = im.depth();
		switch (img_depth) {
			case 0:
				cout << "The data type of current image is CV_8U. \n" << endl;
				break;
			case 1:
				cout << "The data type of current image is CV_8S. \n" << endl;
				break;
			case 2:
				cout << "The data type of current image is CV_16U. \n" << endl;
				break;
			case 3:
				cout << "The data type of current image is CV_16S. \n" << endl;
				break;
			case 4:
				cout << "The data type of current image is CV_32S. \n" << endl;
				break;
			case 5:
				cout << "The data type of current image is CV_32F. \n" << endl;
				break;
			case 6:
				cout << "The data type of current image is CV_64F. \n" << endl;
				break;
			case 7:
				cout << "The data type of current image is CV_USRTYPE1. \n" << endl;
				break;
		}

		cout << "\n show Image depth:\n" << im.depth() << "\n show Image channels :\n " << im.channels() << endl;
		imshow("Image", im);

		double min_v, max_v;
		cv::minMaxLoc(im, &min_v, &max_v);
		cout << "\n show Image min, max:\n" << min_v << "," << max_v << endl;

//		double fstChannel, sndChannel,thrChannel;

		if (static_cast<int>(im.channels()) == 1) {
			//若为灰度图，显示鼠标点击的坐标以及灰度值
			cout << "at(" << position.x() << "," << position.y() << ")value is:"
			     << static_cast<float >(im.at<float>(position.x(), position.y())) << endl;
		} else if (static_cast<int>(im.channels() == 3)) {
			//若图像为彩色图像，则显示鼠标点击坐标以及对应的B, G, R值
			cout << "at (" << position.x() << ", " << position.y() << ")"
			     << "  R value is: " << static_cast<float>(im.at<Vec3f>(position.x(), position.y())[2])
			     << "  G value is: " << static_cast<float>(im.at<Vec3f>(position.x(), position.y())[1])
			     << "  B value is: " << static_cast<float >(im.at<Vec3f>(position.x(), position.y())[0])
			     << endl;
		}

		waitKey(0);

	}


	template<typename T>
	T rotationErr(Eigen::Matrix<T, 3, 3> rotation_gt, Eigen::Matrix<T, 3, 3> rotation_rs) {


		T compare1 = max(acos(std::min(std::max(rotation_gt.col(0).dot(rotation_rs.col(0)), -1.0), 1.0)),
		                 acos(std::min(std::max(rotation_gt.col(1).dot(rotation_rs.col(1)), -1.0), 1.0)));

		return max(compare1, acos(std::min(std::max(rotation_gt.col(2).dot(rotation_rs.col(2)), -1.0), 1.0))) * 180.0 /
		       M_PI;

	}

	template<typename T>
	T translationErr(Eigen::Matrix<T, 3, 1> translation_gt, Eigen::Matrix<T, 3, 1> translation_es) {
		return (translation_gt - translation_es).norm() / translation_gt.norm();
	}


	Scalar depthErr(const Mat &depth_gt, const Mat &depth_es) {

		if (depth_es.depth() != depth_gt.depth()) { std::cerr << "the depth image type are different!" << endl; }
		return cv::sum(cv::abs(depth_gt - depth_es)) / cv::sum(depth_gt);

	}

	template<typename T>
	Eigen::Matrix<T, 3, 3> rotation_pertabation(const T pertabation_x, const T pertabation_y, const T pertabation_z,
	                                            const Eigen::Matrix<T, 3, 3> &Rotation, double &roErr) {

		Eigen::Matrix<T, 3, 3> R;

		T roll = pertabation_x / 180.0 * M_PI;
		T yaw = pertabation_y / 180.0 * M_PI;
		T pitch = pertabation_z / 180.0 * M_PI;


		Eigen::AngleAxis<T> rollAngle(roll, Eigen::Matrix<T, 3, 1>::UnitZ());
		Eigen::AngleAxis<T> yawAngle(yaw, Eigen::Matrix<T, 3, 1>::UnitY());
		Eigen::AngleAxis<T> pitchAngle(pitch, Eigen::Matrix<T, 3, 1>::UnitX());
		Eigen::Quaternion<T> q = rollAngle * yawAngle * pitchAngle;

		R = q.matrix();
		Eigen::Matrix<T, 3, 3> updatedRotation;
		updatedRotation.setZero();
		updatedRotation = R * Rotation;
//		cout<<" ----------------R------------:"<< R<< endl;
//		cout<<" ----------------Eigen::Matrix<T,3,1>::UnitX()-----------:"<< Eigen::Matrix<T,3,1>::UnitX()<< endl;
//		cout<<"Show the rotation loss:"<<updatedRotation<< endl;
		roErr = rotationErr(Rotation, updatedRotation);

		return updatedRotation;
	}

	template<typename T>
	Eigen::Matrix<T, 3, 1> translation_pertabation(const T pertabation_x, const T pertabation_y, const T pertabation_z,
	                                               const Eigen::Matrix<T, 3, 1> &translation, double &roErr) {

		Eigen::Matrix<T, 3, 1> updated_translation;
		updated_translation.setZero();
		updated_translation.x() = translation.x() * (1.0 + pertabation_x);
		updated_translation.y() = translation.y() * (1.0 + pertabation_y);
		updated_translation.z() = translation.z() * (1.0 + pertabation_z);
		roErr = translationErr(translation, updated_translation);
		return updated_translation;

	}

	template<typename T>
	Sophus::SE3<T> posePerturbation(Eigen::Matrix<T, 6, 1> se3, const Sophus::SE3<T> &pose_GT, double &roErr,
	                                const Eigen::Matrix<T, 3, 3> &Rotation, double &trErr,
	                                const Eigen::Matrix<T, 3, 1> &translation) {
		Sophus::SE3<T> SE3_updated = Sophus::SE3<T>::exp(se3) * pose_GT;
		trErr = translationErr(translation, SE3_updated.translation());
		roErr = rotationErr(Rotation, SE3_updated.rotationMatrix());
		return SE3_updated;
	}

	void downscale(Mat &image, const Mat& depth, Eigen::Matrix3f &K, int &level, Mat &image_d, Mat &depth_d, Eigen::Matrix3f &K_d) {

		if (level <= 1) {
			image_d = image;
			// remove negative gray values
			image_d=cv::max(image_d,0.0);
			depth_d = depth;
			// set all nan zero
			Mat mask = Mat(depth_d != depth_d);
			depth_d.setTo(0.0, mask);
			K_d = K;
			return;
		}

		// downscale camera intrinsics

		K_d << K(0, 0) / 2.0, 0, (K(0, 2) + 0.5) / 2.0 - 0.5,
				0, K(1, 1) / 2.0, (K(1, 2) + 0.5) / 2 - 0.5,
				0, 0, 1;
		pyrDown(image, image_d, Size(image.cols / 2, image.rows / 2));
		pyrDown(depth, depth_d, Size(depth.cols / 2, depth.rows / 2));
		// remove negative gray values
		image_d=cv::max(image_d,0.0);
		// set all nan zero
		Mat mask = Mat(depth_d != depth_d);
		depth_d.setTo(0.0, mask);

		level -= 1;
		downscale(image_d, depth_d, K_d, level, image_d, depth_d, K_d);
	}




	void showScaledImage(const Mat & org_GT, const Mat & GT, const Mat& ES){

		double max_orig, min_orig;
		cv::minMaxLoc(GT, &min_orig,&max_orig);
		double max_adj, min_adj;
		cv::minMaxLoc(ES, &min_adj,&max_adj);

//	 double max_real= max(max_adj, max_orig);
//	 double min_real=min(min_adj, min_orig);
		double max_real=0.1;
		double min_real=0.001;



//	 Mat GT_for_show= 25*GT*(1.0/(max_real-min_real))+(-min_real*(1.0/(max_real-min_real)));
//	 Mat ES_for_show= 25*ES*(1.0/(max_real-min_real))+(-min_real*(1.0/(max_real-min_real)));
		Mat GT_for_show= GT*(1.0/(max_real-min_real))+(-min_real*(1.0/(max_real-min_real)));
		Mat ES_for_show= ES*(1.0/(max_real-min_real))+(-min_real*(1.0/(max_real-min_real)));
		Mat GT_orig_for_show= org_GT*(1.0/(max_real-min_real))+(-min_real*(1.0/(max_real-min_real)));

		//org_GT


		imshow("GT_NS_for_show", GT_orig_for_show);
		imshow("GT_for_show", GT_for_show);
		imshow("ES_for_show", ES_for_show);

//	 GT_for_show.convertTo(GT_for_show, CV_32FC1);
//	 ES_for_show.convertTo(ES_for_show, CV_32FC1);
//
//	 imwrite("GT_for_show.exr",GT_for_show); //inv_depth_ref
//	 imwrite("ES_for_show.exr",ES_for_show);

		waitKey(0);



	}

	void projection_K(double uj, double vj,double iDepth,Sophus::SO3d& Rotation,
	                  Eigen::Vector3d& Translation, Eigen::Matrix<double, 2, 1>& pt2d){

		Eigen::Matrix<double,3,3> K;
		K<< 1361.1, 0, 320,
				0, 1361.1, 240,
				0,   0,  1;

		double fx = K(0, 0), cx = K(0, 2), fy =  K(1, 1), cy =K(1, 2);
		Eigen::Matrix<double,3,1> p_3d_no_d;
		p_3d_no_d<< (uj-cx)/fx, (vj-cy)/fy,(double )1.0;
		Eigen::Matrix<double, 3,1> p_c1 ;
		p_c1 <<  p_3d_no_d.x() /iDepth,  p_3d_no_d.y() /iDepth ,p_3d_no_d.z() /iDepth;

		cout<<"show parameter using:<<\n"<<Rotation.matrix()<<","<<Translation<<endl;
		Eigen::Matrix<double, 3, 1> p1 = Rotation * p_c1+Translation ;

//		Eigen::Matrix<double, 3, 1> p1 = p_c1;

		Eigen::Matrix<double,3,1> p_3d_transformed2,point_K;
		p_3d_transformed2=p1;
		point_K = K*p_3d_transformed2;

		pt2d.x()=point_K.x()/point_K.z();
		pt2d.y()=point_K.y()/point_K.z();

	}

	template<typename T>
	bool checkImageBoundaries(const Eigen::Matrix<T, 2, 1>& pixel, int width, int height)
	{
		return (pixel[0] > 1.1 && pixel[0] < width - 2.1 && pixel[1] > 1.1 && pixel[1] < height - 2.1);
	}


	bool project(double uj, double vj, double iDepth, int width, int height,
	             Eigen::Matrix<double, 2, 1>& pt2d,  Sophus::SO3d& Rotation,
	             Eigen::Vector3d& Translation)
	{
		Eigen::Matrix<double,2,1> point_2d_K;
//		cout<<"show parameter later:<<\n"<<Rotation.matrix()<<","<<Translation<<endl;
		projection_K(uj, vj, iDepth,Rotation, Translation, point_2d_K);
		return checkImageBoundaries(point_2d_K, width, height);
	}




}


