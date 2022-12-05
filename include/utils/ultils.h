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
#include <sophus/se3.hpp>


#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/loss_function.h>


#include <iostream>
#include <vector>
#include <cmath>
//#include <omp.h>

#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <chrono>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/search.h>
#include <pcl/surface/mls.h>

#include <pcl/point_types.h>
#include <pcl/io/io.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d.h>


namespace DSONL{

	using namespace cv;
	using namespace std;
	const double DEG_TO_ARC = 0.0174532925199433;


//	void 1dTo2d(){
//	cv::Mat image_ref(hG[i], wG[i], CV_64FC1);
//	memcpy(image_ref.data, newFrame_ref->img_pyr[i], wG[i]*hG[i]*sizeof(float));
//	}


	template<typename T>struct ptsNormal{
	std::vector<Eigen::Matrix<T,3,1>> pts;
	cv::Mat normal_map;
	};

	template<typename T>
	 bool checkImageBoundaries(const Eigen::Matrix<T, 2, 1>& pixel, int width, int height)
	{
		return (pixel[0] > 1.1 && pixel[0] < width - 2.1 && pixel[1] > 1.1 && pixel[1] < height - 2.1);
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
	// out - newIDepth: inverse depth in new image
	// return: if successfully projected or not due to OOB

	void savePointCloud(Mat depth_map,Sophus::SO3d& Rotation,
	                    Eigen::Vector3d& Translation, int  pointCloudIdx){
		pcl::PCDWriter writer;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_right (new pcl::PointCloud<pcl::PointXYZ>);


//		double fov_y= 20;
//		double near= 0.5;
//		double far= 15.0;
//		double aspect= 1.333333;
//		double coeA= 2*far*near/(near-far);
//		double coeB= (far+near)/(near-far);
//		double f= 1.0/(tan(0.5*fov_y* M_PI/180.0)*aspect);

		Eigen::Matrix<double, 4,4> M_new, M_inv;
//		M_new << 1.0/(tan(0.5*fov_y * M_PI/180.0)*aspect), 0, 0, 0,
//				0,  1.0/tan(0.5*fov_y * M_PI/180.0), 0,  0,
//				0,0, (far+near)/(near-far), 2*far*near/(near-far),
//				0,  0,   -1,    0;
//
//
//		M_inv=M_new.inverse();

		// clear up data
		Eigen::Matrix<double,3,3> K;
		K<< 1361.1, 0, 320,
				0, 1361.1, 240,
				0,   0,  1;

		for(int x = 0; x < depth_map.rows; ++x)
		{
			for(int y = 0; y < depth_map.cols; ++y)
			{

//				// M matrix transform and project
//				double d_mapped= 2.0 *1.0/depth_map.at<double>(x,y)-1.0;
//				double x_mapped= 2.0* y/ depth_map.cols -1.0;
//				double y_mapped= 2.0* x/ depth_map.rows -1.0;
//
//				Eigen::Matrix<double,4,1> p_3d, p_2d;
//				p_2d.x()=x_mapped;
//				p_2d.y()=y_mapped;
//				p_2d.z()=d_mapped;
//				p_2d.w()=1.0;
//
//				Eigen::Matrix<double,4,1> p_3d_transformed;
//				p_3d=M_inv*p_2d;
//
//
//				Eigen::Matrix<double, 4,1> projectedPoint;
//////
				Eigen::Matrix<double,3,1> point,p_3d_transformed_K;
//////
//				p_3d_transformed_K.x()= -p_3d.x()/p_3d.w();
//				p_3d_transformed_K.y()= -p_3d.y()/p_3d.w();
//				p_3d_transformed_K.z()= -p_3d.z()/p_3d.w();

				double fx = K(0, 0), cx = K(0, 2), fy =  K(1, 1), cy =K(1, 2);
				Eigen::Matrix<double,3,1> p_3d_no_d;
				p_3d_no_d<< ((double)y-cx)/fx, ((double)x-cy)/fy,(double)1.0;
				Eigen::Matrix<double, 3,1> p_c1 ;
				p_c1 <<  p_3d_no_d.x() /depth_map.at<double>(x,y),  p_3d_no_d.y() /depth_map.at<double>(x,y) ,p_3d_no_d.z() /depth_map.at<double>(x,y);

//				Eigen::Matrix<double, 3, 1> p1 = Rotation * p_c1+Translation ;
//				Eigen::Matrix<double, 3, 1> p1 = p_c1;

//
//				cloud->push_back(pcl::PointXYZ(p_3d_transformed_K.x(), p_3d_transformed_K.y(), p_3d_transformed_K.z()));
//				point = (Rotation*p_3d_transformed_K+Translation);
//				cloud_right->push_back(pcl::PointXYZ(point.x(), point.y(), point.z()));

//				cloud->push_back(pcl::PointXYZ(p_c1.x(), p_c1.y(), p_c1.z()));
					if(pointCloudIdx==1){
						point = (Rotation*p_c1+Translation);

					}else{
						point=p_c1;
					}

				cloud_right->push_back(pcl::PointXYZ(point.x(), point.y(), point.z()));

			}
		}


		if(pointCloudIdx==1){
			writer.write("PointCloud_Transformed__bigBaseline_1.pcd",*cloud_right, false);
		}
		else{
			writer.write("PointCloud_Right_bigBaseline_1.pcd",*cloud_right, false);
		}

//		writer.write("PointCloud_Left_Linear.pcd",*cloud, false);



	};


	void projection_M(double uj, double vj,double iDepth,int width, int height,Sophus::SO3d& Rotation,
	                  Eigen::Vector3d& Translation, Eigen::Matrix<double, 2, 1>& pt2d){
		double fov_y= 20;
		double near= 0.5;
		double far= 15.0;
		double aspect= 1.333333;
		double coeA= 2*far*near/(near-far);
		double coeB= (far+near)/(near-far);
		double f= 1.0/(tan(0.5*fov_y* M_PI/180.0)*aspect);

		Eigen::Matrix<double, 4,4> M_new;
		M_new << 1.0/(tan(0.5*fov_y * M_PI/180.0)*aspect), 0, 0, 0,
				0,  1.0/tan(0.5*fov_y * M_PI/180.0), 0,  0,
				0,0, (far+near)/(near-far), 2*far*near/(near-far),
				0,  0,   -1,    0;

		Eigen::Matrix<double,3,3> K;
		K<< 1361.1, 0, 320,
				0, 1361.1, 240,
				0,   0,  1;


		// transform and project
		double d_mapped= 2.0 *1.0/iDepth-1.0;
		double x_mapped= 2.0* uj/ width -1.0;
		double y_mapped= 2.0* vj/ height -1.0;

		Eigen::Matrix<double,4,1> p_3d, p_2d;
		p_2d.x()=x_mapped;
		p_2d.y()=y_mapped;
		p_2d.z()=d_mapped;
		p_2d.w()=1.0;

		p_3d=M_new.inverse()*p_2d;

		Eigen::Matrix<double,3,1> point,p_3d_transformed_K;

		p_3d_transformed_K.x()= -p_3d.x()/p_3d.w();
		p_3d_transformed_K.y()= -p_3d.y()/p_3d.w();
		p_3d_transformed_K.z()= -p_3d.z()/p_3d.w();

		point = K*(Rotation*p_3d_transformed_K+Translation);
		pt2d.x()=point.x()/point.z();
		pt2d.y()=point.y()/point.z();


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


	bool project(double uj, double vj, double iDepth, int width, int height,
	             Eigen::Matrix<double, 2, 1>& pt2d,  Sophus::SO3d& Rotation,
	             Eigen::Vector3d& Translation)
	{
		Eigen::Matrix<double,2,1> point_2d_K;

//		projection_M(uj, vj, iDepth, width,height,Rotation, Translation, point_2d_M);
		cout<<"show parameter later:<<\n"<<Rotation.matrix()<<","<<Translation<<endl;
		projection_K(uj, vj, iDepth,Rotation, Translation, point_2d_K);
//		project(float (v),float (u), float(depth_ref.at<double>(u,v)),cols_,rows_,KRKi,Kt,pt2d,newIDepth);

		// check image boundaries
		return checkImageBoundaries(pt2d, width, height);
	}



	template<typename T>
	bool project(T uj, T vj, T iDepth, int width, int height,
	             const Eigen::Matrix<T, 3, 3>& KRKinv, const Eigen::Matrix<T, 3, 1>& Kt,
	             Eigen::Matrix<T, 2, 1>& pt2d, T& newIDepth)
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

		// inverse depth in new image
		newIDepth = iDepth*rescale;

		// normalize
		pt2d[0] = pt[0] * rescale;
		pt2d[1] = pt[1] * rescale;

		// check image boundaries
		return checkImageBoundaries(pt2d, width, height);
	}



	bool AddGaussianNoise_Opencv(const Mat mSrc, Mat &mDst,double Mean=0.0, double StdDev=10.0)
	{
		if(mSrc.empty())
		{
			cout<<"[Error]! Input Image Empty!";
			return 0;
		}
//		Mat mSrc_16SC;
//		Mat mGaussian_noise = Mat(mSrc.size(),CV_16SC3);
//		randn(mGaussian_noise,Scalar::all(Mean), Scalar::all(StdDev));
//
//		mSrc.convertTo(mSrc_16SC,CV_16SC3);
//		addWeighted(mSrc_16SC, 1.0, mGaussian_noise, 1.0, 0.0, mSrc_16SC);
//		mSrc_16SC.convertTo(mDst,mSrc.type());
		Mat mSrc_32FC1;
		Mat mGaussian_noise = Mat(mSrc.size(),CV_64FC1);
		randn(mGaussian_noise,Scalar::all(Mean), Scalar::all(StdDev));
//		cout<<"StdDev";

		mSrc.convertTo(mSrc_32FC1,CV_64FC1);
		addWeighted(mSrc_32FC1, 1.0, mGaussian_noise, 1.0, 0.0, mSrc_32FC1);
		mSrc_32FC1.convertTo(mDst,mSrc.type());

		return true;
	}

	bool AddGaussianNoise_Opencv(const Mat mSrc, Mat &mDst,double Mean, double StdDev,  float* statusMap)
	{
		if(mSrc.empty())
		{
			cout<<"[Error]! Input Image Empty!";
			return 0;
		}

		Mat mSrc_32FC1;
		Mat mGaussian_noise = Mat(mSrc.size(),CV_64FC1);
		randn(mGaussian_noise,Scalar::all(Mean), Scalar::all(StdDev));
		mSrc.convertTo(mSrc_32FC1,CV_64FC1);
		for (int u = 0; u< mSrc.rows; u++) // colId, cols: 0 to 480
		{
			for (int v = 0; v < mSrc.cols; v++) // rowId,  rows: 0 to 640
			{
				if (statusMap!=NULL && statusMap[u*mSrc.cols+v]!=0 ){
					mSrc_32FC1.at<double>(u,v)+=mGaussian_noise.at<double>(u,v);

				}
			}
		}
//		addWeighted(mSrc_32FC1, 1.0, mGaussian_noise, 1.0, 0.0, mSrc_32FC1);
		mSrc_32FC1.convertTo(mDst,mSrc.type());

		return true;
	}


	void newNormal(){
		// filtered normal map
//	cv::Mat normal_map_sdf(480, 640, CV_32FC3);
//	cv::Mat nx, ny, nz, med_depth;
//	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
//	NormalEstimator<float>* NEst;
//	NEst = new NormalEstimator<float>(640, 480, K, cv::Size(2*5+1, 2*5+1));
////	NEst = new NormalEstimator<float>(640, 480, K_, cv::Size(2*3+1, 2*3+1));
//
//	NEst->compute(depth_ref,nx, ny, nz);
//	cv::Mat* x0_ptr = NEst->x0_ptr();
//	cv::Mat* y0_ptr = NEst->y0_ptr();
//	cv::Mat* n_sq_inv_ptr = NEst->n_sq_inv_ptr();
//	const float* nx_ptr = (const float*)nx.data;
//	const float* ny_ptr = (const float*)ny.data;
//	const float* nz_ptr = (const float*)nz.data;
//
//	const float* x_hom_ptr = (const float*)x0_ptr->data;
//	const float* y_hom_ptr = (const float*)y0_ptr->data;
//
//	const float* hom_inv_ptr = (const float*)n_sq_inv_ptr->data;
//	const float* z_ptr = (const float*)depth_ref.data;
//	const float* zm_ptr = (const float*)med_depth.data;
//
//	for (int u = 0; u< depth_ref.rows; u++) // colId, cols: 0 to 480
//	{
//		for (int v = 0; v < depth_ref.cols; v++) // rowId,  rows: 0 to 640
//		{
//			const size_t idx = u * depth_ref.cols + v;
//			const float z = z_ptr[idx];
//
////	        if (z <= 0.5 || z >= 65.0 ) // z out of range or unreliable z
////		        continue;
//
//			const Eigen::Vector3f  xy_hom(x_hom_ptr[idx], y_hom_ptr[idx], 1.);
//			const Eigen::Vector3f normal(nx_ptr[idx], ny_ptr[idx], nz_ptr[idx]);
//			if (normal.squaredNorm() < .1) {continue; }
//			if (normal.dot(xy_hom) * normal.dot(xy_hom) * hom_inv_ptr[idx] < .25) // normal direction too far from viewing ray direction (>72.5°)
//				continue;
//
////	        Vec3f d_n_rgb( normal.normalized().z()*0.5+0.5,  normal.normalized().y()*0.5+0.5,  normal.normalized().x()*0.5+0.5);
//	        Vec3f d_n_rgb( normal.normalized().z(),  normal.normalized().y(),  normal.normalized().x());
////			Vec3f d_n_rgb( normal.z(),  normal.y(),  normal.x());
//			normal_map_sdf.at<Vec3f>(u,v)=d_n_rgb;
//
//		}
//	}
//	std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
//	std::chrono::duration<double> time_used =
//			std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
//	cout << "construct the normals: " << time_used.count() << " seconds." << endl;
//
////	imshow("newNormalmap_sdf", normal_map_sdf);
////
////	waitKey(0);
//
//	for (int u = 0; u< depth_ref.rows; u++) // colId, cols: 0 to 480
//	{
//		for (int v = 0; v < depth_ref.cols; v++) // rowId,  rows: 0 to 640
//		{
//
//			Eigen::Vector3d normal_new( normal_map_sdf.at<Vec3f>(u,v)[2],  normal_map_sdf.at<Vec3f>(u,v)[1], normal_map_sdf.at<Vec3f>(u,v)[0]);
//
//			Eigen::Vector3d principal_axis(0, 0, 1);
//			if(normal_new.dot(principal_axis)>0)
//			{
//				normal_new = -normal_new;
//			}
//
//			normal_map.at<Vec3d>(u,v)[0]=normal_new(0);
//			normal_map.at<Vec3d>(u,v)[1]=normal_new(1);
//			normal_map.at<Vec3d>(u,v)[2]=normal_new(2);
//
//		}
//	}
//

	}







    Eigen::Matrix3d rotmatz(double a)
    {
	    Eigen::Matrix3d R;
		R<<cos(a), -sin(a), 0,
		   sin(a) , cos(a), 0,
	         0,     0,      1;
	    return R;
	}
	Eigen::Matrix3d rotmatx(double a)
	{
		Eigen::Matrix3d R;
		R<<     1,     0,   0,
				0,cos(a), -sin(a),
				0,   sin(a) , cos(a);
		return R;
	}

	Eigen::Matrix3d rotmaty(double a)
	{
		Eigen::Matrix3d R;
		R<<      cos(a) ,0,sin(a),
		          0,    1 ,    0,
				-sin(a), 0, cos(a);
		return R;
	}
	Eigen::Vector3d light_C1( Eigen::Vector3d light_w){




//
//		Eigen::Matrix3d R_X=  rotmatx(-3.793*DEG_TO_ARC);
//		Eigen::Matrix3d R_Y=  rotmaty(-178.917*DEG_TO_ARC);
//		Eigen::Matrix3d R_Z=  rotmatz(0*DEG_TO_ARC);

		Eigen::Quaterniond q(0.009445649,-0.0003128,-0.9994076,-0.0330920);





//	    Eigen::Matrix3d R_1w=  R_Y*R_X*R_Z;
//
//		Eigen::Matrix3d R_1w_new = (Eigen::AngleAxisd(-178.917*DEG_TO_ARC, Eigen::Vector3d::UnitY()) *
//		                   Eigen::AngleAxisd(0*DEG_TO_ARC, Eigen::Vector3d::UnitZ()) *
//		                   Eigen::AngleAxisd(-3.793*DEG_TO_ARC, Eigen::Vector3d::UnitX())).toRotationMatrix();




		Eigen::Vector3d t_1w;
		t_1w<<3.8, -16.5, 26.1;

//		Eigen::Matrix3d  R_w1;
//		Eigen::Vector3d  t_w1;
//		R_w1=R_1w.transpose();
//        R_w1=(R_Y*R_X*R_Z).transpose();

//		t_w1= - R_1w.transpose()*t_1w;
//		t_w1=- (R_Y*R_X*R_Z).transpose()*t_1w;


//		return (R_w1* light_w+ t_w1);

		return (q.toRotationMatrix()).transpose()* (light_w -t_1w);

	}






   class normalMapFiltering {
   private:
	   // member function to pad the image before convolution
	   Mat padding(Mat img, int k_width, int k_height, string type) {
		   Mat scr;
		   img.convertTo(scr, CV_64FC1);
		   int pad_rows, pad_cols;
		   pad_rows = (k_height - 1) / 2;
		   pad_cols = (k_width - 1) / 2;
		   Mat pad_image(Size(scr.cols + 2 * pad_cols, scr.rows + 2 * pad_rows), CV_64FC1, Scalar(0));
		   scr.copyTo(pad_image(Rect(pad_cols, pad_rows, scr.cols, scr.rows)));
		   // mirror padding
		   if (type == "mirror") {
			   for (int i = 0; i < pad_rows; i++) {
				   scr(Rect(0, pad_rows - i, scr.cols, 1)).copyTo(pad_image(Rect(pad_cols, i, scr.cols, 1)));
				   scr(Rect(0, (scr.rows - 1) - pad_rows + i, scr.cols, 1)).copyTo(pad_image(Rect(pad_cols,
				                                                                                  (pad_image.rows - 1) -
				                                                                                  i, scr.cols, 1)));
			   }

			   for (int j = 0; j < pad_cols; j++) {
				   pad_image(Rect(2 * pad_cols - j, 0, 1, pad_image.rows)).copyTo(
						   pad_image(Rect(j, 0, 1, pad_image.rows)));
				   pad_image(Rect((pad_image.cols - 1) - 2 * pad_cols + j, 0, 1, pad_image.rows)).
						   copyTo(pad_image(Rect((pad_image.cols - 1) - j, 0, 1, pad_image.rows)));
			   }

			   return pad_image;
		   }
			   // replicate padding
		   else if (type == "replicate") {
			   for (int i = 0; i < pad_rows; i++) {
				   scr(Rect(0, 0, scr.cols, 1)).copyTo(pad_image(Rect(pad_cols, i, scr.cols, 1)));
				   scr(Rect(0, (scr.rows - 1), scr.cols, 1)).copyTo(pad_image(Rect(pad_cols,
				                                                                   (pad_image.rows - 1) - i, scr.cols,
				                                                                   1)));
			   }

			   for (int j = 0; j < pad_cols; j++) {
				   pad_image(Rect(pad_cols, 0, 1, pad_image.rows)).copyTo(pad_image(Rect(j, 0, 1, pad_image.rows)));
				   pad_image(Rect((pad_image.cols - 1) - pad_cols, 0, 1, pad_image.rows)).
						   copyTo(pad_image(Rect((pad_image.cols - 1) - j, 0, 1, pad_image.rows)));
			   }
			   // zero padding
			   return pad_image;
		   } else {
			   return pad_image;
		   }

	   }


	   // member function to define kernels for convolution
	   Mat define_kernel(int k_width, int k_height, string type)
	   {
		   // box kernel
		   if (type == "box")
		   {
			   Mat kernel(k_height, k_width, CV_64FC1, Scalar(1.0 / (k_width * k_height)));
			   return kernel;
		   }
			   // gaussian kernel
		   else if (type == "gaussian")
		   {
			   // I will assume k = 1 and sigma = 1
			   int pad_rows = (k_height - 1) / 2;
			   int pad_cols = (k_width - 1) / 2;
			   Mat kernel(k_height, k_width, CV_64FC1);
			   for (int i = -pad_rows; i <= pad_rows; i++)
			   {
				   for (int j = -pad_cols; j <= pad_cols; j++)
				   {
					   kernel.at<double>(i + pad_rows, j + pad_cols) = exp(-(i*i + j*j) / 2.0);
				   }
			   }

			   kernel = kernel /sum(kernel).val[0];
			   return kernel;
		   }
	   }

   public:

	   void convolve(Mat scr, Mat &dst, int k_w, int k_h, string paddingType, string filterType)
	   {
		   Mat pad_img, kernel;
		   pad_img = padding(scr, k_w, k_h, paddingType);
		   kernel = define_kernel(k_w, k_h, filterType);

		   Mat output = Mat::zeros(scr.size(), CV_64FC1);

		   for (int i = 0; i < scr.rows; i++)
		   {
			   for (int j = 0; j < scr.cols; j++)
			   {
				   output.at<double>(i, j) = sum(kernel.mul(pad_img(Rect(j, i, k_w, k_h)))).val[0];
			   }
		   }

//		   output.convertTo(dst, CV_8UC1);
			dst=output;
	   }

   };




	// show image function
	void imageInfo(Mat im_,  Eigen::Vector2i& position ){

		//position,  fst is rowIdx, snd is colIdx
		Mat im=im_ ;
       int img_depth= im.depth();
		switch (img_depth) {
			case 0:cout<<"The data type of current image is CV_8U. \n"<<endl;
				break;
			case 1:cout<<"The data type of current image is CV_8S. \n"<<endl;
				break;
			case 2:cout<<"The data type of current image is CV_16U. \n"<<endl;
				break;
			case 3:cout<<"The data type of current image is CV_16S. \n"<<endl;
				break;
			case 4:cout<<"The data type of current image is CV_32S. \n"<<endl;
				break;
			case 5:cout<<"The data type of current image is CV_32F. \n"<<endl;
				break;
			case 6:cout<<"The data type of current image is CV_64F. \n"<<endl;
				break;
			case 7:cout<<"The data type of current image is CV_USRTYPE1. \n"<<endl;
				break;
		}

		cout<<"\n show Image depth:\n"<<im.depth()<<"\n show Image channels :\n "<< im.channels()<<endl;
		imshow("Image", im);

		double min_v, max_v;
		cv::minMaxLoc(im, &min_v, &max_v);
		cout<<"\n show Image min, max:\n"<<min_v<<","<<max_v<<endl;

//		double fstChannel, sndChannel,thrChannel;

		if (static_cast<int>(im.channels())==1)
		{
			//若为灰度图，显示鼠标点击的坐标以及灰度值
			cout<<"at("<<position.x()<<","<<position.y()<<")value is:"<<static_cast<float >(im.at<float>(position.x(),position.y()))<<endl;
		}
		else if (static_cast<int>(im.channels() == 3))
		{
			//若图像为彩色图像，则显示鼠标点击坐标以及对应的B, G, R值
			cout << "at (" << position.x() << ", " << position.y() << ")"
			     << "  R value is: " << static_cast<float>(im.at<Vec3f>(position.x(), position.y())[2])
				 << "  G value is: " << static_cast<float>(im.at<Vec3f>(position.x(), position.y())[1])
				 << "  B value is: " << static_cast<float >(im.at<Vec3f>(position.x(), position.y())[0])
			     << endl;
		}

		waitKey(0);

	}














	template<typename T> Eigen::Matrix<T,3,1> backProjection(const Eigen::Matrix<T,2,1> &pixelCoord, const Eigen::Matrix<T,3,3> & K, const Mat& depthMap ){
        // convert depth map into grid2D table
		cv::Mat flat_depth_map = depthMap.reshape(1, depthMap.total() * depthMap.channels());
		std::vector<T> img_depth_values=depthMap.isContinuous() ? flat_depth_map : flat_depth_map.clone();
		std::unique_ptr<ceres::Grid2D<T> > grid2d_depth;
		std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<T>> >BiCubicInterpolator_depth;

		grid2d_depth.reset(new ceres::Grid2D<double>(&img_depth_values[0],0, depthMap.rows, 0, depthMap.cols));
		BiCubicInterpolator_depth.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<T>>(*grid2d_depth));

		// get depth value at pixelCoord(col, row)
		T u, v , d;
		u= (T) pixelCoord(1);
		v= (T) pixelCoord(0);
		BiCubicInterpolator_depth->Evaluate(u,v,&d);
		T fx = K(0, 0), cx = K(0, 2), fy =  K(1, 1), cy = K(1, 2);
		// back projection
		Eigen::Matrix<T,3,1> p_3d_no_d;
		p_3d_no_d<< (v-cx)/fx, (u-cy)/fy,1.0;
		Eigen::Matrix<T, 3,1> p_c1=d*p_3d_no_d;
		return  p_c1;

	}


	Mat singleNormalFiltering(Mat& normalMap, int k_w, int k_h ){

		Mat newNormalsMap(normalMap.rows, normalMap.cols, CV_64FC3, Scalar(0,0,0)); // B,G,R

		vector<Mat> channels_new, channels_result;
		Mat normal_z, normal_y,normal_x, normal_z_new, normal_y_new, normal_x_new;
		Mat channels[3];
		split(normalMap,channels);
		normal_z= channels[0];
		normal_y= channels[1];
		normal_x= channels[2];


		normalMapFiltering normalFilter;

		normalFilter.convolve(normal_z,normal_z_new, k_w, k_h, "zero", "box"); // zero ,mirror ,  replicate ;   gaussian, box
		normalFilter.convolve(normal_y,normal_y_new, k_w, k_h, "zero", "box");
		normalFilter.convolve(normal_x,normal_x_new, k_w, k_h, "zero", "box");

		Mat norm_mat;
		sqrt((normal_z_new.mul(normal_z_new)+ normal_y_new.mul(normal_y_new) + normal_x_new.mul(normal_x_new)), norm_mat);

		channels_result.push_back(normal_z_new/norm_mat);
		channels_result.push_back(normal_y_new/norm_mat);
		channels_result.push_back(normal_x_new/norm_mat);
		merge(channels_result,newNormalsMap);

		return  newNormalsMap;

	}

	Mat normalMapFilter(Mat& normalMap){

		Mat newNormalsMap(normalMap.rows, normalMap.cols, CV_64FC3, Scalar(0,0,0)); // B,G,R
		newNormalsMap= singleNormalFiltering(normalMap, 5,5);
//		Mat normalFilter2 = singleNormalFiltering(newNormalsMap, 7, 7);
//		imshow("normals", normalMap);
//		imshow("first filter 5*5", newNormalsMap);
//		imshow(" second filter 7*7", normalFilter2);
//		waitKey(0);

		return newNormalsMap;

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


	Eigen::Vector3d backprojection_realDepth(double& depth, int& p_row, int& p_col, Eigen::Matrix<double,3,3>& K_){

		double fx = K_(0, 0), cx = K_(0, 2), fy =  K_(1, 1), cy = K_(1, 2), f=30.0;
		Eigen::Matrix<double,3,1> p_3d_no_d;
		p_3d_no_d<< (p_col-cx)/fx, (p_row-cy)/fy,1.0;
		Eigen::Matrix<double, 3,1> p_c1=depth*p_3d_no_d;

		return  p_c1;

	}


//
//	int iterate_all_pts(int n_eigen_current, const std::vector<Eigen::Vector3d>& cloud_eigen, Eigen::Vector3d pt_mls,
//	                    bool& is_match) // n_eigen is a bigger one
//	{
//		int match_n_eigen = 0;
//
//		for(int n_eigen = n_eigen_current; n_eigen < cloud_eigen.size(); n_eigen++)
//		{
//			Eigen::Vector3d pt_eigen = cloud_eigen[n_eigen];
//
//
//
//			Eigen::Vector3d vec_diff = pt_eigen - pt_mls;
//			if(vec_diff.norm() < 0.1)
////        if(vec_diff.norm() < 0.05)
//			{
//				match_n_eigen = n_eigen;
//				is_match = true;
//				break;
//			}
//
//		}
//
//		if(is_match == false)
//		{
//			std::cout << "fail to match !!!!!!!!" << std::endl;
//			match_n_eigen = n_eigen_current;
//		}
//
//
//
//		return match_n_eigen;
//	}
//
//
//	void resamplePts_and_compNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_init, std::vector<Eigen::Vector3d> cloud_eigen, cv::Mat init_normal_map)
//	{
//
//		// Create a KD-Tree
//		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
//
//		// Output has the PointNormal type in order to store the normals calculated by MLS
//		pcl::PointCloud<pcl::PointNormal> mls_points;
//
//		// Init object (second point type is for the normals, even if unused)
//		pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
//
//		mls.setComputeNormals (true);
//
//		// Set parameters
//		mls.setInputCloud (cloud_init);
//		mls.setPolynomialOrder (2);
//		mls.setSearchMethod (tree);
////    mls.setSearchRadius (0.1); // 304511 pts
////    mls.setSearchRadius (0.15); // 306776 pts
////        mls.setSearchRadius (0.2); // 307073 pts
//		mls.setSearchRadius (0.5); // 307200 pts
//
//
//
//		// Reconstruct
//		mls.process (mls_points);
//
////	std::cout << "number of new pts:" << std::endl;
////	std::cout << mls_points.size() << std::endl;
//
//
//		int n_eigen_current = 0;
//
//		for(int nIndex = 0; nIndex < mls_points.points.size(); nIndex++)
//		{
//
//
//			//
//			double n_x = mls_points.points[nIndex].normal_x;
//			double n_y = mls_points.points[nIndex].normal_y;
//			double n_z = mls_points.points[nIndex].normal_z;
//
//			Eigen::Vector3d normal_new(n_x, n_y, n_z);
//			normal_new = normal_new.normalized();
//
//			Eigen::Vector3d principal_axis(0, 0, 1);
//			if(normal_new.dot(principal_axis)<0)
//			{
//				normal_new = -normal_new;
//			}
//
//			//
//			double pt_x = mls_points.points[nIndex].x;
//			double pt_y = mls_points.points[nIndex].y;
//			double pt_z = mls_points.points[nIndex].z;
//
//			Eigen::Vector3d pt_mls(pt_x, pt_y, pt_z);
//
//			bool is_match = false;
//			n_eigen_current = iterate_all_pts(n_eigen_current, cloud_eigen, pt_mls, // assign pt_mls's normal to eigen map's pixel
//			                                  is_match);
//
//			if(is_match == false)
//			{
//				std::cout << "use orig value !!!!!!!!" << std::endl;
//				continue;
//			}
//
//
////        std::cout << "successful matching" << std::endl;
//
//			int row_id = trunc(n_eigen_current/640); // 640 pixels in a row
//			int col_id = (n_eigen_current - row_id*640) % 640;
//
//			init_normal_map.at<cv::Vec3d>(row_id, col_id)[0] = normal_new(2);
//			init_normal_map.at<cv::Vec3d>(row_id, col_id)[1] = normal_new(1);
//			init_normal_map.at<cv::Vec3d>(row_id, col_id)[2] = normal_new(0);
//
//			n_eigen_current = n_eigen_current + 1;
//
//		}
//
//		std::cout << "====== complete! =======" << std::endl;
//
////	cv::imshow("img3", init_normal_map);
////	cv::waitKey(0);
//
//
//	}
//
//	void comp_accurate_normals(std::vector<Eigen::Vector3d> cloud_eigen, cv::Mat init_normal_map)
//	{
//		// convert format
//		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//		cloud->points.resize(cloud_eigen.size());
//		for(int i=0; i<cloud_eigen.size(); i++)
//		{
//			cloud->points[i].getVector3fMap() = Eigen::Vector3f(cloud_eigen[i](0), cloud_eigen[i](1), cloud_eigen[i](2)).cast<float>();
//
//		}
//
//		// resample
//		resamplePts_and_compNormal(cloud, cloud_eigen, init_normal_map);
//
//		std::cout << "---------" << std::endl;
//	}
//
//




	void getPtsNormalMap(const Eigen::Matrix<double,3,3> & K_, const Mat& depth,  ptsNormal<double>& pN_ptr){

		Mat normalsMap(depth.rows, depth.cols, CV_64FC3, Scalar(0,0,0)); // B,G,R
		Mat normalsMap_bgr(depth.rows, depth.cols, CV_64FC3, Scalar(0,0,0)); // B,G,R
		double fx = K_(0, 0), cx = K_(0, 2), fy =  K_(1, 1), cy = K_(1, 2), f=30.0;
		// focal length: 30
		cout <<"fx:"<<fx <<"fy:"<<fy<<endl;
		std::unordered_map<int, int> inliers_filter;

		for(int x = 0; x < depth.rows; ++x)
		{
			for(int y = 0; y < depth.cols; ++y)
			{
				double d= depth.at<double>(x,y);
				Eigen::Matrix<double,3,1> p_3d_no_d;
				p_3d_no_d<< (y-cx)/fx, (x-cy)/fy,1.0;
				Eigen::Matrix<double, 3,1> p_c1=d*p_3d_no_d;

				pN_ptr.pts.push_back(p_c1);
				double d_x1= depth.at<double>(x,y+1);
				double  d_y1= depth.at<double>(x+1, y);
				// calculate normal for each point
				Eigen::Matrix<double, 3,1> normal, v_x, v_y;
				v_x <<  ((d_x1-d)*(y-cx)+d_x1)/fx, (d_x1-d)*(x-cy)/fy , (d_x1-d);
				v_y << (d_y1-d)*(y-cx)/fx,(d_y1+ (d_y1-d)*(x-cy))/fy, (d_y1-d);
				v_x=v_x.normalized();
				v_y=v_y.normalized();
//				normal=v_y.cross(v_x);// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111
				normal=v_x.cross(v_y);
				normal=normal.normalized();
				Vec3d d_n(normal.z(), normal.y(), normal.x());
				normalsMap.at<Vec3d>(x, y) = d_n;
			}
		}
				pN_ptr.normal_map=normalsMap;


	}

	Mat getNormals(const Eigen::Matrix<double,3,3> & K_, const Mat& depth){

		Mat normalsMap(depth.rows, depth.cols, CV_64FC3, Scalar(0,0,0)); // B,G,R
		Mat normalsMap_bgr(depth.rows, depth.cols, CV_64FC3, Scalar(0,0,0)); // B,G,R
		double fx = K_(0, 0), cx = K_(0, 2), fy =  K_(1, 1), cy = K_(1, 2), f=30.0;
		// focal length: 30
		cout <<"fx:"<<fx <<"fy:"<<fy<<endl;
		std::unordered_map<int, int> inliers_filter;


//		inliers_filter.emplace(229, 335); //yes
//		inliers_filter.emplace(232, 333); //yes
//		inliers_filter.emplace(234, 335); //yes


		for(int x = 0; x < depth.rows; ++x)
		{
			for(int y = 0; y < depth.cols; ++y)
			{
//				if(inliers_filter.count(x)==0){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//		        if(inliers_filter[x]!=y ){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//				cout<<" \n show the coordinates:"<<x<<","<<y<<"---> value:"<<depth.at<double>(x,y)<<endl;
				double d= depth.at<double>(x,y);
				Eigen::Matrix<double,3,1> p_3d_no_d;
				p_3d_no_d<< (y-cx)/fx, (x-cy)/fy,1.0;
				Eigen::Matrix<double, 3,1> p_c1=d*p_3d_no_d;
				double d_x1= depth.at<double>(x,y+1);
				double  d_y1= depth.at<double>(x+1, y);
				// calculate normal for each point
				Eigen::Matrix<double, 3,1> normal, v_x, v_y;
				v_x <<  ((d_x1-d)*(y-cx)+d_x1)/fx, (d_x1-d)*(x-cy)/fy , (d_x1-d);
				v_y << (d_y1-d)*(y-cx)/fx,(d_y1+ (d_y1-d)*(x-cy))/fy, (d_y1-d);
				v_x=v_x.normalized();
				v_y=v_y.normalized();
				normal=v_y.cross(v_x);// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111
//				normal=v_x.cross(v_y);
				normal=normal.normalized();

//				Vec3d d_n_rgb(normal.x(),normal.y(),normal.z());
//				Vec3d d_n_rgb(normal.z()*0.5+0.5, normal.y()*0.5+0.5, normal.x()*0.5+0.5);
//				Vec3d d_n(normal.z(), normal.y(), normal.x());
//				normalsMap_bgr.at<Vec3d>(x, y) = d_n_rgb;
				Vec3d d_n(normal.x(), normal.y(), normal.z());
				normalsMap.at<Vec3d>(x, y) = d_n;
			}
		}

		// normal map filtering

//	    Mat normalMapafterfilter= normalMapFilter(normalsMap);// when checking result, uncommit it

//		Mat normalMapafterfilter_channel[3];
//		vector<Mat> channels_result;
//
//		split(normalMapafterfilter, normalMapafterfilter_channel);
//
//		Mat normal_zafterfilter= normalMapafterfilter_channel[0]*0.5+0.5;
//		Mat normal_yafterfilter= normalMapafterfilter_channel[1]*0.5+0.5;
//		Mat normal_xafterfilter= normalMapafterfilter_channel[2]*0.5+0.5;
//
//		channels_result.push_back(normal_zafterfilter);
//		channels_result.push_back(normal_yafterfilter);
//		channels_result.push_back(normal_xafterfilter);
//
//
//		Mat normalMapafterfilter_result;
//		merge(channels_result, normalMapafterfilter_result);
//
//
//		imshow("normalsMap_1", normalsMap_bgr);
//		waitKey(0);
//		return normalMapFilter(normalsMap);
		return normalsMap;


	}



	Mat getNormals_renderedDepth(const Eigen::Matrix<double,4,4> & M, const Mat& depth){

		Mat normalsMap(depth.rows, depth.cols, CV_64FC3, Scalar(0,0,0)); // B,G,R
		Mat normalsMap_bgr(depth.rows, depth.cols, CV_64FC3, Scalar(0,0,0)); // B,G,R
		double fov_y= 33.398;
		double near= 0.01;
		double far= 60.0;
		double aspect= 1.333;
		double coeA= 2*far*near/(near-far);
		double coeB= (far+near)/(near-far);
		double f= 1.0/(tan(0.5*fov_y)*aspect);


		for(int x = 0; x < depth.rows; ++x)
		{
			for(int y = 0; y < depth.cols; ++y)
			{

				double d_mapped= 2.0 *depth.at<double>(x,y)-1.0;
				double x_mapped= 2.0* x/ depth.rows -1.0;
				double y_mapped= 2.0* y/ depth.cols -1.0;

//				Eigen::Vector4d p_h(y_mapped,x_mapped,d_mapped,1.0);
//				Eigen::Vector4d projPointInCam= M.inverse()*p_h;
//				Eigen::Matrix<double,3,1> p_3d;
//				p_3d<< projPointInCam.x()/projPointInCam.w(),  projPointInCam.y()/projPointInCam.w(), projPointInCam.z()/projPointInCam.w();

				Eigen::Matrix<double,3,1> p_3d;
				p_3d.z()=-coeA/(d_mapped+coeB);
				p_3d.x()=y_mapped*aspect*(-p_3d.z())/f;
				p_3d.y()=x_mapped*(-p_3d.z())/f;

				double d_x1= 2.0 *depth.at<double>(x,y+1)-1.0;
				double y_1mapped= 2.0* (y+1)/ 640.0 -1.0;
//				Eigen::Vector4d p_h_1(y_1mapped, x_mapped,d_x1,1.0);
//				Eigen::Vector4d projPointInCam1= M.inverse()*p_h_1;
				Eigen::Matrix<double,3,1> p_3d1;
//				p_3d1<< projPointInCam1.x()/projPointInCam1.w(),  projPointInCam1.y()/projPointInCam1.w(), projPointInCam1.z()/projPointInCam1.w();
				p_3d1.z()=-coeA/(d_x1+coeB);
				p_3d1.x()=y_1mapped*aspect*(-p_3d1.z())/f;
				p_3d1.y()=x_mapped*(-p_3d1.z())/f;




				double  d_y1=  2.0 *depth.at<double>(x+1, y)-1.0;
				double x_1mapped= 2.0* (x+1)/ depth.rows -1.0;

//				Eigen::Vector4d p_h_2(y_mapped, x_1mapped,d_y1,1.0);
//				Eigen::Vector4d projPointInCam2= M.inverse()*p_h_1;
				Eigen::Matrix<double,3,1> p_3d2;
//				p_3d2<< projPointInCam2.x()/projPointInCam2.w(),  projPointInCam2.y()/projPointInCam2.w(), projPointInCam2.z()/projPointInCam2.w();

				p_3d2.z()=-coeA/(d_y1+coeB);
				p_3d2.x()=y_mapped*aspect*(-p_3d2.z())/f;
				p_3d2.y()=x_1mapped*(-p_3d2.z())/f;

				Eigen::Vector3d v_x(p_3d1-p_3d);
				Eigen::Vector3d v_y(p_3d2-p_3d);


				Eigen::Vector3d normal;
				v_x=v_x.normalized();
				v_y=v_y.normalized();
				normal=v_x.cross(v_y);
				normal=normal.normalized();


				Vec3d d_n_rgb(normal.z()*0.5+0.5, normal.y()*0.5+0.5, normal.x()*0.5+0.5);
//				Vec3d d_n(normal.z(), normal.y(), normal.x());
				normalsMap_bgr.at<Vec3d>(x, y) = d_n_rgb;
//				normalsMap.at<Vec3d>(x, y) = d_n;
			}
		}

		imshow("normalsMap2", normalsMap_bgr);
		waitKey(0);
		return normalMapFilter(normalsMap_bgr);


	}

	template<typename T>
	T rotationErr(Eigen::Matrix<T, 3,3> rotation_gt, Eigen::Matrix<T, 3,3> rotation_rs){


		T compare1= max( acos(  std::min(std::max(rotation_gt.col(0).dot(rotation_rs.col(0)), -1.0), 1.0) ) , acos( std::min(std::max(rotation_gt.col(1).dot(rotation_rs.col(1)), -1.0), 1.0)  ) );

		return max(compare1, acos( std::min(std::max(rotation_gt.col(2).dot(rotation_rs.col(2)), -1.0), 1.0)  ) ) * 180.0/ M_PI;

	}
	template<typename T>
	T translationErr(Eigen::Matrix<T, 3,1> translation_gt, Eigen::Matrix<T, 3,1> translation_es){
		return (translation_gt-translation_es).norm() / translation_gt.norm() ;
	}


	Scalar depthErr( const Mat& depth_gt, const Mat& depth_es ){

		if (depth_es.depth()!=depth_gt.depth()){std::cerr<<"the depth image type are different!"<<endl;}
		return cv::sum(cv::abs(depth_gt-depth_es)) / cv::sum(depth_gt);

	}

	template<typename T>
	Eigen::Matrix<T,3,3> rotation_pertabation(const T pertabation_x,const T pertabation_y, const T pertabation_z, const Eigen::Matrix<T,3,3>& Rotation, double& roErr){

		Eigen::Matrix<T,3,3> R;

		T roll= pertabation_x/180.0 *M_PI;
		T yaw=pertabation_y /180.0 *M_PI;
		T pitch= pertabation_z/180.0 *M_PI;


		Eigen::AngleAxis<T> rollAngle(roll, Eigen::Matrix<T,3,1>::UnitZ());
		Eigen::AngleAxis<T> yawAngle(yaw, Eigen::Matrix<T,3,1>::UnitY());
		Eigen::AngleAxis<T> pitchAngle(pitch, Eigen::Matrix<T,3,1>::UnitX());
		Eigen::Quaternion<T> q = rollAngle * yawAngle * pitchAngle;

		R=q.matrix();
//		R=Eigen::AngleAxis<T>(pertabation_x/180.0 *M_PI,Eigen::Matrix<T,3,1>::UnitX())
//							    * Eigen::AngleAxis<T>(pertabation_y/180.0*M_PI,   Eigen::Matrix<T,3,1>::UnitY())
//							    * Eigen::AngleAxis<T>(pertabation_z/180.0*M_PI,  Eigen::Matrix<T,3,1>::UnitZ());

		Eigen::Matrix<T,3,3> updatedRotation;
		updatedRotation.setZero();
		updatedRotation= R*Rotation;
//		cout<<" ----------------R------------:"<< R<< endl;
//		cout<<" ----------------Eigen::Matrix<T,3,1>::UnitX()-----------:"<< Eigen::Matrix<T,3,1>::UnitX()<< endl;
//		cout<<"Show the rotation loss:"<<updatedRotation<< endl;
		roErr=rotationErr(Rotation, updatedRotation);

		return updatedRotation;
	}
	template<typename T>
	Eigen::Matrix<T,3,1> translation_pertabation(const T pertabation_x,const T pertabation_y, const T pertabation_z, const Eigen::Matrix<T,3,1>& translation, double& roErr){

		Eigen::Matrix<T,3,1> updated_translation;
		updated_translation.setZero();
		updated_translation.x()= translation.x()*(1.0+pertabation_x);
		updated_translation.y()= translation.y()*(1.0+pertabation_y);
		updated_translation.z()= translation.z()*(1.0+pertabation_z);

		roErr=translationErr(translation, updated_translation);
		return updated_translation;



	}

	template<typename T>
	Sophus::SE3<T> posePerturbation(Eigen::Matrix<T,6,1> se3, const Sophus::SE3<T>& pose_GT , double& roErr, const Eigen::Matrix<T,3,3>& Rotation, double& trErr,const Eigen::Matrix<T,3,1>& translation){

		Sophus::SE3<T> SE3_updated= Sophus::SE3<T>::exp(se3)*pose_GT;


		trErr=translationErr(translation, SE3_updated.translation());
		roErr= rotationErr(Rotation, SE3_updated.rotationMatrix());

		return SE3_updated;



	}











	Mat colorMap(Mat& deltaMap, float upper, float lower){
		Mat deltaMap_Color(deltaMap.rows, deltaMap.cols, CV_32FC3, Scalar(0,0,0));


		for(int x = 0; x < deltaMap.rows; ++x) {
			for (int y = 0; y < deltaMap.cols; ++y) {

				float delta= deltaMap.at<float>(x,y);
				if (delta< upper && delta> lower){

					delta=delta*(1.0/(upper-lower))+(-lower*(1.0/(upper-lower)));
					deltaMap_Color.at<Vec3f>(x,y)[0]=delta;
					deltaMap_Color.at<Vec3f>(x,y)[1]=delta;
					deltaMap_Color.at<Vec3f>(x,y)[2]=delta;

				} else if(delta>upper){
					deltaMap_Color.at<Vec3f>(x,y)[0]=0;
					deltaMap_Color.at<Vec3f>(x,y)[1]=1;
					deltaMap_Color.at<Vec3f>(x,y)[2]=0;
				}else if( delta<lower && delta!=-1){
					deltaMap_Color.at<Vec3f>(x,y)[0]=1;
					deltaMap_Color.at<Vec3f>(x,y)[1]=0;
					deltaMap_Color.at<Vec3f>(x,y)[2]=1;
				}
				else if(delta==-1){
					deltaMap_Color.at<Vec3f>(x,y)[0]=0;
					deltaMap_Color.at<Vec3f>(x,y)[1]=0;
					deltaMap_Color.at<Vec3f>(x,y)[2]=1;

				}else{
					deltaMap_Color.at<Vec3f>(x,y)[0]=1;
					deltaMap_Color.at<Vec3f>(x,y)[1]=0;
					deltaMap_Color.at<Vec3f>(x,y)[2]=0;
				}


			}
		}

		return deltaMap_Color;
	}

//	void minusMap(Mat& Img_left,  Mat& Img_right, Mat& deltaMap){
//
//		Mat deltaMapGT(depth_left.rows, depth_left.cols, CV_32FC1, Scalar(-1)); // default value 1.0
////		Mat deltaMapGT(depth_left.rows, depth_left.cols, CV_32FC1, Scalar(1)); // default value 1.0
//		double fx = K_(0, 0), cx = K_(0, 2), fy =  K_(1, 1), cy = K_(1, 2);
//		pcl::PCDWriter writer;
//		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
//		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rig (new pcl::PointCloud<pcl::PointXYZ>);
//
//
//		for(int x = 0; x < depth_left.rows; ++x) {
//			for (int y = 0; y < depth_left.cols; ++y) {
//				double d_r= depth_right.at<double>(x,y);
//				Eigen::Matrix<double,3,1> p_3d_no_d_r;
//				p_3d_no_d_r<< (y-cx)/fx, (x-cy)/fy,1.0;
//				Eigen::Matrix<double, 3,1> p_c2=d_r*p_3d_no_d_r;
//				cloud_rig->push_back(pcl::PointXYZ(p_c2.x(), p_c2.y(), p_c2.z()));
//			}
//		}
//
//		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
//		kdtree.setInputCloud (cloud_rig);
//
//		std::vector<int> pointIdxRadiusSearch;
//		std::vector<float> pointRadiusSquaredDistance;
//
//		float radius =thres;
//
//		for(int x = 0; x < depth_left.rows; ++x)
//		{
//			for(int y = 0; y < depth_left.cols; ++y)
//			{
////
////				if(inliers_filter.count(x)==0){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
////				if(inliers_filter[x]!=y ){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~
//
//
//				// calculate 3D point of left camera
//				double d= depth_left.at<double>(x,y);
//				Eigen::Matrix<double,3,1> p_3d_no_d;
//				p_3d_no_d<< (y-cx)/fx, (x-cy)/fy,1.0;
//				Eigen::Matrix<double, 3,1> p_c1=d*p_3d_no_d;
//
//				Eigen::Vector3d  point_Trans=ExtrinsicPose.rotationMatrix() *  p_c1+ExtrinsicPose.translation();
//
//				pcl::PointXYZ searchPoint(point_Trans.x(),point_Trans.y(),point_Trans.z());
//				if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
//				{
////					for (std::size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
////						std::cout << "\n------"  <<   (*cloud_rig)[ pointIdxRadiusSearch[0] ].x
////						          << " " << (*cloud_rig)[ pointIdxRadiusSearch[0] ].y
////						          << " " << (*cloud_rig)[ pointIdxRadiusSearch[0] ].z
////						          << " (squared distance: " << pointRadiusSquaredDistance[0] << ")" << std::endl;
////
//
//					double left_intensity=Img_left.at<double>(x,y);
//					float pointCorres_x= (*cloud_rig)[ pointIdxRadiusSearch[0] ].x;
//					float pointCorres_y= (*cloud_rig)[ pointIdxRadiusSearch[0] ].y;
//					float pointCorres_z= (*cloud_rig)[ pointIdxRadiusSearch[0] ].z;
//					float pixel_x=(fx*pointCorres_x)/pointCorres_z+cx;
//					float pixel_y= (fy*pointCorres_y)/pointCorres_z+cy;
//					float right_intensity=Img_right.at<double>(round(pixel_y), round(pixel_x));
//
//
//
//	}
//

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

    void showMinus(Mat& minus_original, Mat& minus_adjust, Mat& minus_mask ){

		double max_orig, min_orig;
		cv::minMaxLoc(minus_original, &min_orig,&max_orig);
	    double max_adj, min_adj;
	    cv::minMaxLoc(minus_adjust, &min_adj,&max_adj);

		double max_real= max(max_adj, max_orig);
		double min_real=min(min_adj, min_orig);

	    Mat adj_show(minus_original.rows, minus_original.cols, CV_32FC3, Scalar(0,0,0));
	    Mat orig_show(minus_original.rows, minus_original.cols, CV_32FC3, Scalar(0,0,0));


	    for(int x = 0; x < minus_original.rows; ++x) {
		    for (int y = 0; y < minus_original.cols; ++y) {
			    if (minus_mask.at<uchar>(x,y)==1){

				   float minus_orig_val= minus_original.at<float>(x,y);

				    minus_orig_val*=(1.0/(max_real-min_real))+(-min_real*(1.0/(max_real-min_real)));

				    float minus_adj_val= minus_adjust.at<float>(x,y);
				    minus_adj_val*=(1.0/(max_real-min_real))+(-min_real*(1.0/(max_real-min_real)));
				    if (isnan(minus_orig_val) || isnan(minus_adj_val)){
					    orig_show.at<Vec3f>(x,y)[0]=1;
					    orig_show.at<Vec3f>(x,y)[1]=0;
					    orig_show.at<Vec3f>(x,y)[2]=0;

					    adj_show.at<Vec3f>(x,y)[0]=1;
					    adj_show.at<Vec3f>(x,y)[1]=0;
					    adj_show.at<Vec3f>(x,y)[2]=0;
					    continue;
					}
				    orig_show.at<Vec3f>(x,y)[0]=minus_orig_val;
				    orig_show.at<Vec3f>(x,y)[1]=minus_orig_val;
				    orig_show.at<Vec3f>(x,y)[2]=minus_orig_val;

				    adj_show.at<Vec3f>(x,y)[0]=minus_adj_val;
				    adj_show.at<Vec3f>(x,y)[1]=minus_adj_val;
				    adj_show.at<Vec3f>(x,y)[2]=minus_adj_val;

				}else{

				    orig_show.at<Vec3f>(x,y)[0]=0;
				    orig_show.at<Vec3f>(x,y)[1]=0;
				    orig_show.at<Vec3f>(x,y)[2]=1;

				    adj_show.at<Vec3f>(x,y)[0]=0;
				    adj_show.at<Vec3f>(x,y)[1]=0;
				    adj_show.at<Vec3f>(x,y)[2]=1;

				}
			}
	    }

	    imshow("orig_show",orig_show);
	    imshow("adj_show",adj_show);

	    imwrite("orig_show.png",orig_show);
	    imwrite("adj_show.png",adj_show);

//	    waitKey(0);
	}

	Mat deltaMapGT(Mat& Img_left,  Mat& depth_left,  Mat& Img_right, Mat& depth_right,const Eigen::Matrix<double,3,3> & K_, double & thres,
				   const  Sophus::SE3d & ExtrinsicPose, float& upper,float& buttom, Mat& pred_deltaMap ){

//		Mat normalMap_left=getNormals(K_,depth_left);
//		Mat normalMap_right=getNormals(K_,depth_right);

		Mat deltaMapGT(depth_left.rows, depth_left.cols, CV_32FC1, Scalar(-1)); // default value 1.0
//		Mat deltaMapGT(depth_left.rows, depth_left.cols, CV_32FC1, Scalar(1)); // default value 1.0
		double fx = K_(0, 0), cx = K_(0, 2), fy =  K_(1, 1), cy = K_(1, 2);
		pcl::PCDWriter writer;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rig (new pcl::PointCloud<pcl::PointXYZ>);


		for(int x = 0; x < depth_left.rows; ++x) {
			for (int y = 0; y < depth_left.cols; ++y) {
				double d_r= depth_right.at<double>(x,y);
				Eigen::Matrix<double,3,1> p_3d_no_d_r;
				p_3d_no_d_r<< (y-cx)/fx, (x-cy)/fy,1.0;
				Eigen::Matrix<double, 3,1> p_c2=d_r*p_3d_no_d_r;
				cloud_rig->push_back(pcl::PointXYZ(p_c2.x(), p_c2.y(), p_c2.z()));
			}
		}

		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
		kdtree.setInputCloud (cloud_rig);
		double  max, min;
		cv::minMaxLoc(Img_left, &min, &max);
		cout<<"\n show max and min of Img_left:\n"<< max <<","<<min<<endl;
		cv::minMaxLoc(Img_right, &min, &max);
		cout<<"\n show max and min of Img_right:\n"<< max <<","<<min<<endl;
		std::unordered_map<int, int> inliers_filter;
		//new image

//		inliers_filter.emplace(173,333); //yes
		inliers_filter.emplace(213,295); //yes

		std::vector<int> pointIdxRadiusSearch;
		std::vector<float> pointRadiusSquaredDistance;

		float radius =thres;

		Mat minus_original(depth_left.rows, depth_left.cols, CV_32FC1, Scalar(0));
		Mat minus_adjust(depth_left.rows, depth_left.cols, CV_32FC1, Scalar(0));
		Mat minus_mask(depth_left.rows, depth_left.cols, CV_8UC1,Scalar(0));

		for(int x = 0; x < depth_left.rows; ++x)
		{
			for(int y = 0; y < depth_left.cols; ++y)
			{
//
//				if(inliers_filter.count(x)==0){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//				if(inliers_filter[x]!=y ){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~
				// calculate 3D point of left camera
				double d= depth_left.at<double>(x,y);
				Eigen::Matrix<double,3,1> p_3d_no_d;
				p_3d_no_d<< (y-cx)/fx, (x-cy)/fy,1.0;
				Eigen::Matrix<double, 3,1> p_c1=d*p_3d_no_d;

				Eigen::Vector3d  point_Trans=ExtrinsicPose.rotationMatrix() *  p_c1+ExtrinsicPose.translation();

				pcl::PointXYZ searchPoint(point_Trans.x(),point_Trans.y(),point_Trans.z());
				if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
				{
//					for (std::size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
//						std::cout << "\n------"  <<   (*cloud_rig)[ pointIdxRadiusSearch[0] ].x
//						          << " " << (*cloud_rig)[ pointIdxRadiusSearch[0] ].y
//						          << " " << (*cloud_rig)[ pointIdxRadiusSearch[0] ].z
//						          << " (squared distance: " << pointRadiusSquaredDistance[0] << ")" << std::endl;
//
					float left_intensity=Img_left.at<double>(x,y);
					float pointCorres_x= (*cloud_rig)[ pointIdxRadiusSearch[0] ].x;
					float pointCorres_y= (*cloud_rig)[ pointIdxRadiusSearch[0] ].y;
					float pointCorres_z= (*cloud_rig)[ pointIdxRadiusSearch[0] ].z;
					float pixel_x=(fx*pointCorres_x)/pointCorres_z+cx;
					float pixel_y= (fy*pointCorres_y)/pointCorres_z+cy;
					float right_intensity=Img_right.at<double>(round(pixel_y), round(pixel_x));
					float delta=right_intensity/left_intensity;
                    //float delta= abs(left_intensity-right_intensity);

					float diff_orig=std::abs(left_intensity-right_intensity);
					minus_original.at<float>(x,y)=diff_orig;
					float delta_pred=pred_deltaMap.at<float>(x,y);
					float diff_adj=std::abs(left_intensity-delta_pred*right_intensity);
					minus_adjust.at<float>(x,y)=diff_adj;
					minus_mask.at<uchar>(x,y)=1;

					deltaMapGT.at<float>(x,y)=delta;


				}


//				double left_intensity=Img_left.at<double>(x,y);
//				for(int x_ = 0; x_ < depth_left.rows; ++x_) {
//					for (int y_ = 0; y_ < depth_left.cols; ++y_) {
//
//						double d_r= depth_right.at<double>(x_,y_);
//						Eigen::Matrix<double,3,1> p_3d_no_d_r;
//						p_3d_no_d_r<< (y_-cx)/fx, (x_-cy)/fy,1.0;
//						Eigen::Matrix<double, 3,1> p_c2=d_r*p_3d_no_d_r;
//						cloud_rig->push_back(pcl::PointXYZ(p_c2.x(), p_c2.y(), p_c2.z()));
//						double distance= ((point_Trans-p_c2).norm());
//						if ( distance< thres){
//							double rigth_intensity=Img_right.at<double>(x_,y_);
//							deltaMapGT.at<double>(x,y)=left_intensity/rigth_intensity;
//						}
//					}
//				}
//
//
//				double d_r= depth_right.at<double>(x,y);
//				Eigen::Matrix<double,3,1> p_3d_no_d_r;
//				p_3d_no_d_r<< (y-cx)/fx, (x-cy)/fy,1.0;
//				Eigen::Matrix<double, 3,1> p_c2=d_r*p_3d_no_d_r;
//				cloud_rig->push_back(pcl::PointXYZ(p_c2.x(), p_c2.y(), p_c2.z()));

			}
		}

//		imwrite("red_mask.png", minus_mask); // ---------------------------------just save the red mask-------------------------
		showMinus(minus_original,minus_adjust, minus_mask);

//		writer.write("PointCloud_Transformed.pcd",*cloud, false);//
//		writer.write("PointCloud_right_HD.pcd",*cloud_rig, false);//

		double  max_n, min_n;
		cv::minMaxLoc(deltaMapGT, &min_n, &max_n);
//		deltaMapGT=deltaMapGT*(1.0/(upper-buttom))+(-buttom*(1.0/(upper-buttom)));
		cout<<"\n show max and min of deltaMapGT:\n"<< max_n <<","<<min_n<<endl;


		return deltaMapGT;
	}








	float bilinearInterpolation(const Mat &image, const float &x, const float &y) {
		const int x1 = floor(x), x2 = ceil(x), y1 = floor(y), y2 = ceil(y);

		int width = image.cols, height = image.rows;

		//两个差值的中值
		float f12, f34;
		float epsilon = 0.0001;
		//四个临近像素坐标x像素值
		float f1, f2, f3, f4;

		if ((x < 0) || (x > width - 1) || (y < 0) || (y > height - 1)) {
			return -1.0;
		} else {
			if (fabs(x - width + 1) <= epsilon) { //如果计算点在右测边缘

				//如果差值点在图像的最右下角
				if (fabs(y - height + 1) <= epsilon) {
					f1 = image.at<float>(x1, y1);
					return f1;
				} else {
					f1 = image.at<float>(x1, y1);
					f3 = image.at<float>(x1, y2);

					//图像右方的插值
					return ((float) (f1 + (y - y1) * (f3 - f1)));
				}
			} else if (fabs(y - height + 1) <= epsilon) {
				f1 = image.at<float>(x1, y1);
				f2 = image.at<float>(x2, y1);
				return ((float) (f1 + (x - x1) * (f2 - f1)));
			} else {
				//得计算四个临近点像素值
				f1 = image.at<float>(x1, y1);
				f2 = image.at<float>(x2, y1);
				f3 = image.at<float>(x1, y2);
				f4 = image.at<float>(x2, y2);

				//第一次插值
				f12 = f1 + (x - x1) * (f2 - f1); // f(x,0)

				//第二次插值
				f34 = f3 + (x - x1) * (f4 - f3); // f(x,1)

				//最终插值
				return ((float) (f12 + (y - y1) * (f34 - f12)));
			}
		}
	}

	void printAll(const double *arr, int n) {
		cout << "show value of n:" << n << endl;
		for (int i = 0; i < n / 10; i++) {
			cout << arr[i];
			cout << ((i + 1) % 20 ? ' ' : '\n');
		}
	}

	bool removeNegativeValue(Mat& src, Mat& dst){
		dst = cv::max(src, 0);
		return true;
	}


	void MLS(){

		//	for (int u = 0; u< depth_ref.rows; u++) // colId, cols: 0 to 480
//	{
//		for (int v = 0; v < depth_ref.cols; v++) // rowId,  rows: 0 to 640
//		{
//
//			double d=depth_ref.at<double>(u,v);
//			double d_x1= depth_ref.at<double>(u,v+1);
//			double d_y1= depth_ref.at<double>(u+1, v);
//
//			// calculate 3D point coordinate
//			Eigen::Vector2d pixelCoord((double)v,(double)u);//  u is the row id , v is col id
//			Eigen::Vector3d p_3d_no_d((pixelCoord(0)-cx)/fx, (pixelCoord(1)-cy)/fy,1.0);
//			Eigen::Vector3d p_c1=d*p_3d_no_d;
//
//			pts.push_back(p_c1);
//			Eigen::Matrix<double,3,1> normal, v_x, v_y;
//			v_x <<  ((d_x1-d)*(v-cx)+d_x1)/fx, (d_x1-d)*(u-cy)/fy , (d_x1-d);
//			v_y << (d_y1-d)*(v-cx)/fx,(d_y1+ (d_y1-d)*(u-cy))/fy, (d_y1-d);
//			v_x=v_x.normalized();
//			v_y=v_y.normalized();
//            normal=v_y.cross(v_x);
////			normal=v_x.cross(v_y);
//			normal=normal.normalized();
//
//			normal_map.at<cv::Vec3d>(u, v)[0] = normal(0);
//			normal_map.at<cv::Vec3d>(u, v)[1] = normal(1);
//			normal_map.at<cv::Vec3d>(u, v)[2] = normal(2);
//
//		}
//	}
//	comp_accurate_normals(pts, normal_map);



	}

	void DrawHist(Mat& src, float& upper_bound, float & lower_bound , string& name){
		std::vector<Mat> bgr_planes;
		split( src, bgr_planes );
		int histSize = 256;
		float range[] = { lower_bound, upper_bound }; //the upper boundary is exclusive
		const float* histRange[] = { range };
		bool uniform = true, accumulate = false;
		Mat b_hist, g_hist, r_hist;
		calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate );
		calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate );
		calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate );
		int hist_w = 640, hist_h = 480;
		int bin_w = cvRound( (double) hist_w/histSize );
		Mat histImage( hist_h, hist_w, CV_32FC3, Scalar( 0,0,0) );
		normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
		normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
		normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
		for( int i = 1; i < histSize; i++ )
		{
			line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ),
			      Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
			      Scalar( 255, 0, 0), 2, 8, 0  );
			line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ),
			      Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
			      Scalar( 0, 255, 0), 2, 8, 0  );
			line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ),
			      Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
			      Scalar( 0, 0, 255), 2, 8, 0  );
		}

//		imshow("Source image", src );
//        string img_name = std::to_string(src.at<Vec3f>(6,6)[0]);
		imshow(name, histImage );
//		waitKey(0);
	}





}


