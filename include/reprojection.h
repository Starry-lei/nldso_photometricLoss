//
// Created by cheng on 13.09.22.
//
#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/loss_function.h>
#include <ceres/rotation.h>

#include <memory>
#include <visnav/local_parameterization_se3.hpp>
#include "ultils.h"
#include "brdfMicrofacet.h"

#include <opencv2/core/eigen.hpp>


namespace DSONL{

	struct PhotometricBAOptions {
		/// 0: silent, 1: ceres brief report (one line), 2: ceres full report
		int verbosity_level = 1;

		/// update intrinsics or keep fixed
		bool optimize_depth = false;

		bool optimize_pose = true;

		/// use huber robust norm or squared norm
		bool use_huber = true;

		/// parameter for huber loss (in pixel)
		double huber_parameter = 4/255.0;

		bool lambertianCase= false;

		bool usePixelSelector= false;

		bool useFilterController= true;

		/// maximum number of solver iterations
		int max_num_iterations = 20;
	};




	template <typename T>
	Eigen::Matrix<T, 2, 1> project( Eigen::Matrix<T, 3, 1> & p, const T& fx , const T& fy,const T& cx ,const T& cy )  {

		 T x = p(0);
		 T y = p(1);
		 T z = p(2);

		Eigen::Matrix<T, 2, 1> res;
		res.setZero();
		if (z > T(0)) {

			 T pixel_x= fx * x / z + cx;
			 T pixel_y= fy * y / z + cy;

//			if(pixel_x> T(0) && pixel_x< cols && pixel_y>T(0) && pixel_y<rows ){
				res.x()=pixel_y;
				res.y()=pixel_x;
//			}
		}
		return res;
	}


	// This function transforms a pixel from the reference image to a new one.
	// It also checks image boundaries and inverse depth consistency even if
	// its values is < 0.
	//
	// in - uj: pixel x coordinate
	// in - vj: pixel y coordinate
	// in - iDepth: pixel inverse depth
	// in - width, height: image dimensions
	// in - KRKinv: K*rotation*inv(K) from reference to new image
	// in - Kt: K*translation from reference to new image
	// out - pt2d: projected point in new image
	// return: if successfully projected or not due to OOB
	template<typename T>
	bool projectRT(T uj, T vj, T iDepth, int width, int height,
	             const Eigen::Matrix<T, 3, 3>& KRKinv, const Eigen::Matrix<T, 3, 1>& Kt,
	             Eigen::Matrix<T, 2, 1>& pt2d)
	{
		// transform and project
		const Eigen::Matrix<T, 3, 1> pt = KRKinv * Eigen::Matrix<T, 3, 1>(uj, vj, 1) + Kt*iDepth;

		// rescale factor
		const T rescale = 1 / pt[2];

		// if the point was in the range [0, Inf] in camera1
		// it has to be also in the same range in camera2
		// This allows using negative inverse depth values
		// i.e. same iDepth sign in both cameras
		if (!(rescale > 0)) return false;

		// normalize
		pt2d[0] = pt[0] * rescale;
		pt2d[1] = pt[1] * rescale;

		// check image boundaries
		return checkImageBoundaries(pt2d, width, height);
	}


	struct GetPixelGrayValue {

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		GetPixelGrayValue(
				const Eigen::Vector2d &pixelCoor,
				const Eigen::Matrix3d & K,
				const int rows,
				const int cols,
				ceres::Grid2D<double>& grid2d_grayImage_right ,
//                          ceres::BiCubicInterpolator<ceres::Grid2D<double>>& interpolator_depth,
				const Mat& gray_Image_ref,
				const Mat& deltaMap


		) {
			rows_ = rows;
			cols_ = cols;
			pixelCoor_ = pixelCoor;
			K_ = K;
			gray_Image_ref_=gray_Image_ref;
			deltaMap_=deltaMap;
			delta_val=deltaMap_.at<double>(pixelCoor_(1),pixelCoor_(0));
			gray_Image_ref_val=gray_Image_ref_.at<double>(pixelCoor_(1),pixelCoor_(0));

			get_pixel_gray_val = std::make_unique<ceres::BiCubicInterpolator<ceres::Grid2D<double> >>(grid2d_grayImage_right);


		}

		template<typename T>
		bool operator()(
				const T* const sT,
				const T* const sd,
				T *residual
		) const {

			Eigen::Map<Sophus::SE3<T> const> const Tran(sT);
//			Eigen::Map<Eigen::Matrix<T,1,1> const> const depth(sd);

			T d, u_, v_, intensity_image_ref, delta;
			intensity_image_ref=(T) gray_Image_ref_val;
//			double delta_falg;
//			delta_falg=delta_val;
			delta=(T)delta_val;


			u_=(T)pixelCoor_(1);
			v_=(T)pixelCoor_(0);

			// unproject
			T fx = (T)K_(0, 0), cx = (T)K_(0, 2), fy =  (T)K_(1, 1), cy = (T)K_(1, 2);
			Eigen::Matrix<T,3,1> p_3d_no_d;
			p_3d_no_d.setZero();
			p_3d_no_d.x()= (v_-cx)/fx;
			p_3d_no_d.y()= (u_-cy)/fy;
			p_3d_no_d.z()= (T)1.0;
//			p_3d_no_d<< (v_-cx)/fx, (u_-cy)/fy,1.0;
//			d= (T) depth_val;
			d=sd[0];
			Eigen::Matrix<T, 3,1> p_c1=sd[0]*p_3d_no_d;

			Eigen::Matrix<T, 3, 1> p1 = Tran * p_c1 ;


			// project
			Eigen::Matrix<T, 2, 1> pt = project(p1,fx, fy,cx, cy);

			if(pt.y()> T(0) && pt.y()<  (T)cols_ && pt.x()>T(0) && pt.x()< (T)rows_ ){
				T pixel_gray_val_out;
				get_pixel_gray_val->Evaluate(pt.x(), pt.y(), &pixel_gray_val_out);
				residual[0] = delta*intensity_image_ref - pixel_gray_val_out;
				return true;

			}


		}

		Mat deltaMap_;
		Mat gray_Image_ref_;
		double rows_, cols_;
		Eigen::Vector2d pixelCoor_;
		Eigen::Matrix3d K_;
		Sophus::SE3<double> CurrentT_;
		double delta_val;
		double gray_Image_ref_val;

		std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> get_pixel_gray_val;



	};



	struct PhotometricCostFunctor {

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		PhotometricCostFunctor(
				const Eigen::Vector2d &pixelCoor,
				const Eigen::Matrix3f & K,
				const Eigen::Matrix3f & Kinv,
				const int rows,
				const int cols,
				const std::vector<double> &vec_pixel_gray_values,
				const double pixel_gray_val_in[9],
				const double delta_val_in[9]
//				const double pixel_gray_val_in[16],
//				const double delta_val_in[16]
		) {
			rows_ = rows;
			cols_ = cols;
			pixelCoor_ = pixelCoor;
			K_ = K.cast<double>();
			Kinv_=Kinv.cast<double>();


			for (int i = 0; i < 9; ++i) {
				gray_Image_ref_val[i]=pixel_gray_val_in[i];
				delta_val[i]=delta_val_in[i];
			}
//			for (int i = 0; i < 16; ++i) {
//				gray_Image_ref_val[i]=pixel_gray_val_in[i];
//				delta_val[i]=delta_val_in[i];
//			}


			grid2d.reset(new ceres::Grid2D<double>(&vec_pixel_gray_values[0], 0, rows_, 0, cols_));
			get_pixel_gray_val.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<double> >(*grid2d));
		}

		template<typename T>
		bool operator()(
				const T* const sRotation,
				const T* const sTranslation,
				const T* const sd,
				T *residual
		) const {

//			Eigen::Map<Eigen::Matrix<T,3,3> const> const Rotation(sRotation);
			Eigen::Map<Sophus::SO3<T> const> const Rotation(sRotation);

//			Eigen::Map<Sophus::SE3<T> const> const Tran(sT);

			Eigen::Map<Eigen::Matrix<T,3,1> const> const Translation(sTranslation);

			T u_, v_;// delta; //intensity_image_ref
//			intensity_image_ref=(T) gray_Image_ref_val;
//			delta=(T)delta_val;
			u_=(T)pixelCoor_(1);
			v_=(T)pixelCoor_(0);

			Eigen::Matrix<T, 2, 1> pt2d;
			// transform and project
			Eigen::Matrix<T, 3, 1> ptd1= Kinv_* Eigen::Matrix<T, 3, 1>(v_, u_, T(1));

			Eigen::Matrix<T, 3, 1> pt = K_*(Rotation*ptd1 + Translation *sd[0]); // convert K into  T type?

			// rescale factor
			 T rescale = T(1) / pt[2];

			// normalize
			pt2d[0] = pt[0] * rescale;
			pt2d[1] = pt[1] * rescale;




//			// unproject
//			T fx = (T)K_(0, 0), cx = (T)K_(0, 2), fy =  (T)K_(1, 1), cy = (T)K_(1, 2);
//			Eigen::Matrix<T,3,1> p_3d_no_d;
//			p_3d_no_d<< (v_-cx)/fx, (u_-cy)/fy,(T)1.0;
//
//			Eigen::Matrix<T, 3,1> p_c1 ;
//			p_c1 <<  p_3d_no_d.x() /sd[0],  p_3d_no_d.y() /sd[0] ,p_3d_no_d.z() /sd[0];
//			Eigen::Matrix<T, 3, 1> p1 = Rotation * p_c1+Translation ;
//			// project
//			Eigen::Matrix<T, 2, 1> pt = project(p1,fx, fy,cx, cy);

//			T res1;
//			T pixel_gray_val_out1;

			for (int i = 0; i < 9; ++i) {

				int m = i / 3;
				int n = i % 3;

				T pixel_gray_val_out, u_l, v_l;

				u_l=pt2d.y()+T(m-1);
				v_l=pt2d.x()+T(n-1);
//				u_l=pt.x()+T(m-1);
//				v_l=pt.y()+T(n-1);
				get_pixel_gray_val->Evaluate(u_l, v_l, &pixel_gray_val_out);
				residual[i] =  T(delta_val[i])* T(gray_Image_ref_val[i]) - pixel_gray_val_out;
			}
//			for (int i = 0; i < 16; ++i) {
//
//				int m = i / 4;
//				int n = i % 4;
//
//				T pixel_gray_val_out, u_l, v_l;
//
//				u_l=pt.x()+T(m-2);
//				v_l=pt.y()+T(n-2);
//				get_pixel_gray_val->Evaluate(u_l, v_l, &pixel_gray_val_out);
//				residual[i] =  T(delta_val[i])* T(gray_Image_ref_val[i]) - pixel_gray_val_out;
//			}
			return true;
		}
		Mat deltaMap_;
		Mat gray_Image_ref_;
		double rows_, cols_;
		Eigen::Vector2d pixelCoor_;
		Eigen::Matrix3d K_;
		Eigen::Matrix3d Kinv_;
		Sophus::SE3<double> CurrentT_;
		double delta_val[9];
		double gray_Image_ref_val[9];
//		double delta_val[16];
//		double gray_Image_ref_val[16];
		std::unique_ptr<ceres::Grid2D<double> > grid2d;
		std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> get_pixel_gray_val;
	};






}