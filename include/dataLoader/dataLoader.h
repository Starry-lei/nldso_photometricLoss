//
// Created by cheng on 10.10.22.
//

#pragma once

#include "settings/setting.h"
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <unordered_map>

#include "PFMReadWrite.h"
#include <iostream>


#define baseline_l 0;
#define baseline_s 1;

namespace DSONL {
	using namespace cv;
	using namespace std;

	struct dataOptions {
		/// 0: big baseline
		/// 1: small baseline
		/// 2: smaller baseline(no specular light on the wall, i.e, lower metallic on wall)
		/// 3: MicroBaseline
		/// 4:  Control Experiment for Lambertian Data  Non-Lambertian Data
		int baseline = 0;
		/// is textured or not
		bool isTextured = true;
		/// use red channel for testing
		int channelIdx = 1; // int channelIdx = 1;
		bool lambertian = true;
		bool remove_outlier_manually = true;
		/// should we calculate 3 channel delta map for loss function now?????????????????
	};

	class dataLoader {

	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		dataLoader();

		~dataLoader();

		dataOptions options_;
        Mat image_ref_metallic;
        Mat image_ref_roughness;
		Mat grayImage_ref;
		Mat grayImage_target;
		Mat depth_map_ref;
		Mat depth_map_target;
		Mat outlier_mask_big_baseline;
		Eigen::Matrix3f camera_intrinsics;
		Eigen::Matrix4f M_matrix;
		Mat normal_map_GT;
		Mat image_ref_baseColor;
		Mat image_target_baseColor;

		Eigen::Matrix3d R12;
		Eigen::Vector3d t12;
		Eigen::Quaterniond q_12;
		Eigen::Matrix3d R1;
		Eigen::Matrix3d R2;
        Sophus::SE3d camPose1;
		int rows;
		int cols;


		void Init() {
			// Camera intrinsics
//			camera_intrinsics <<    1361.1, 0, 320,
//			                        0, 1361.1, 240,
//			                        0, 0, 1;

            camera_intrinsics <<   577.8705, 0, 320,
                                    0, 577.8705, 240,
                                    0, 0, 1;
			Eigen::Matrix3d S_x;
			S_x << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0;

			if (options_.isTextured) {

				string image_ref_path;
				string image_ref_baseColor_path;
				string depth_ref_path;
                string image_ref_metallic_path;
                string image_ref_roughness_path;
                string image_normal_GT_path;

//                image_ref_path =            "../data/SimulationEnvData/leftImage/origfov_10.pfm";
//                image_ref_baseColor_path =  "../data/SimulationEnvData/leftImage/bgrbaseColor_10.pfm";
//                depth_ref_path =            "../data/SimulationEnvData/leftImage/origfovdepth_10.png";
//                image_ref_metallic_path=    "../data/SimulationEnvData/leftImage/bgrmetallic_10.pfm";
//                image_ref_roughness_path =  "../data/SimulationEnvData/leftImage/bgrroughness_10.pfm";

                image_ref_path =            "../data/SimulationEnvData/leftImage/bgr_10.png";
                depth_ref_path =            "../data/SimulationEnvData/leftImage/origfovdepth_10.png";

//                image_ref_baseColor_path =  "../data/SimulationEnvData/leftImage/DSNL/bgrbaseColor_10.pfm";
                image_ref_baseColor_path =  "../data/SimulationEnvData/leftImage/DSNL/bgrbaseColor_10.png";
                image_ref_metallic_path  =    "../data/SimulationEnvData/leftImage/DSNL/metallic_10.pfm";
                image_ref_roughness_path =  "../data/SimulationEnvData/leftImage/DSNL/roughness_10.pfm";
                image_normal_GT_path =      "../data/SimulationEnvData/leftImage/DSNL/normals10.data"; //normals11  normals10 normals9  //data/SimulationEnvData/leftImage/DSNL/normal_10.data

				Eigen::Matrix3d R1_w_l, R1_w_r;// left-handed and right-handed
				Eigen::Vector3d t1_w_l;


                t1_w_l << -1.010461, 0.110693, -1.495948;
                Eigen::Quaternion<double> quaternionR1(  0.059711, 0.244452 ,-0.252040 ,0.934427);
				R1 = quaternionR1.toRotationMatrix();

                // get extrinsic of camera 1
                camPose1.setQuaternion(quaternionR1);
                camPose1.translation()=t1_w_l;

//                cout<<"show camPose1:"<<camPose1.matrix()<<endl;


                Mat image_ref = cv::imread(image_ref_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
                Mat depth_ref = imread(depth_ref_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
                Mat depth_reference(depth_ref.rows, depth_ref.cols, CV_64FC1);
                for (int j = 0; j < depth_ref.rows; ++j) {
                    for (int i = 0; i < depth_ref.cols; ++i) {
                        depth_reference.at<double>(j,i)= 1.0/5000.0 * ((double ) depth_ref.at<unsigned short >(j,i));
//                        cout << "\n show  depth_reference: " << depth_reference.at<double>(j,i)<<endl;
                    }
                }

//				image_ref_baseColor = loadPFM(image_ref_baseColor_path);
                image_ref_baseColor =imread(image_ref_baseColor_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
                image_ref_baseColor.convertTo(image_ref_baseColor, CV_32FC3, 1.0/255.0);

                Mat matallic_C1= loadPFM(image_ref_metallic_path);
                Mat roughness_C1= loadPFM(image_ref_roughness_path);


				int channelIdx = options_.channelIdx;
				extractChannel(image_ref, grayImage_ref, channelIdx);
                grayImage_ref.convertTo(grayImage_ref, CV_64FC1, 1/255.0);

				rows = image_ref.rows;
				cols = image_ref.cols;
				//setGlobalCalib(cols,rows,camera_intrinsics);
				// ref image depth
				Mat channel[3], metallic_ref_render, channel_rough[3], _tar_render;
                image_ref_metallic=matallic_C1;
                image_ref_roughness= roughness_C1;
                depth_map_ref = depth_reference.clone();

                float normalArray[480][640][3]={0.0f};
                ifstream readIn(image_normal_GT_path, ios::in | ios::binary);
                readIn.read((char*) &normalArray, sizeof normalArray);
                cv::Mat normal_A(480,640,CV_32FC3, &normalArray);
                normal_map_GT = normal_A.clone();

//                normal_map_GT = loadPFM(image_normal_GT_path);
//                depth_map_ref.convertTo(depth_map_ref, CV_64FC1);
				// -------------------------------------------Target image data loader-------------------------
				string image_target_path;
				string image_target_baseColor_path;
				string depth_target_path;

				if (options_.baseline == 0) {

					image_target_path = "../data/SimulationEnvData/rightImage/bgr_16.png";
					depth_target_path = "../data/SimulationEnvData/rightImage/origfovdepth_16.png";

					Eigen::Matrix3d R2_w_l, R1_w_r, R2_w_r;
					Eigen::Vector3d t2_w_l;

                    t2_w_l << -0.999663, 0.136382 ,-1.479749;

                    Eigen::Quaterniond quaternionR2( 0.045509, 0.247733, -0.224787, 0.941291);
					R2 = quaternionR2.toRotationMatrix();
					R12 = R2.transpose() * R1;
					t12 = R2.transpose() * (t1_w_l - t2_w_l);
					q_12 = R12;
				}else if (options_.baseline == 1){
					image_target_path = "../data/Env_light/right02/image_rightRGB0803.png";
					depth_target_path = "../data/Env_light/right02/image_rightDepth0803.pfm";

					Eigen::Matrix3d R2_w_l, R1_w_r, R2_w_r;
					Eigen::Vector3d t2_w_l;

//					R2_w_l << -0.945025,  0.022402,  -0.326230 ,
//					        -0.326998,  -0.064742,  0.942805 ,
//					        -0.000000,  0.997651,  0.068508;


					t2_w_l << -1.000, 2.8900, 0.2100;

//					R2_w_r = R2_w_l * S_x;

					Eigen::Quaterniond quaternionR2(R2_w_r);
					R2 = quaternionR2.toRotationMatrix();
					R12 = R2.transpose() * R1;
					t12 = R2.transpose() * (t1_w_l - t2_w_l);
					q_12 = R12;

				}



                Mat image_target= imread(image_target_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
                        //loadPFM(image_target_path);


                Mat depth_target = imread(depth_target_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);


                Mat depth_tar(depth_target.rows, depth_target.cols, CV_64FC1);

                for (int j = 0; j < depth_target.rows; ++j) {
                    for (int i = 0; i < depth_target.cols; ++i) {
                        depth_tar.at<double>(j,i)= 1.0/5000.0 * ((double) depth_target.at<unsigned short >(j,i));
//                        cout << "\n show  depth_target: " << depth_tar.at<float>(j,i)<<endl;
                    }
                }

//				image_target_baseColor = imread(image_target_baseColor_path, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
//				image_target_baseColor.convertTo(image_target_baseColor, CV_64FC3);
				extractChannel(image_target, grayImage_target, channelIdx);
//              cout<<"show grayImage_target depth()"<<grayImage_target.depth()<<endl;

                grayImage_target.convertTo(grayImage_target, CV_64FC1, 1/255.0);

                depth_map_target = depth_tar.clone();
//				depth_map_target.convertTo(depth_map_target, CV_64FC1);


			} else {
				// RGB image without texture
				string image_ref_path = "../data/rgb/No_Texture_Images/rt_16_4_56_cam1_notexture.exr";
				string image_ref_baseColor_path = "../data/rgb/No_Texture_Images/rt_16_5_47_cam1_notexture_basecolor.exr";
				if (options_.baseline == 0) {
					string image_target_path = "../data/rgb/No_Texture_Images/rt_16_4_56_cam5_notexture.exr";
					string image_target_baseColor = "../data/rgb/No_Texture_Images/rt_16_5_47_cam5_notexture_basecolor.exr";
				} else if (options_.baseline == 1) {
					// TODO: small baseline data
					string image_target_path = "../data/rgb/No_Texture_Images/rt_16_4_56_cam5_notexture.exr";
				}
			}
		}
	};

	dataLoader::dataLoader(void) { cout << "The game is loading ..." << endl; }
	dataLoader::~dataLoader(void) { cout << "The program ends here, have a nice day!" << endl; }


}// namespace DSONL

//---------------------------------------------------some notes----------------------------------------------


//R1_w_r = R1_w_l * S_x;
//R1_w_l << -0.970705, 0.029789, -0.238420,
//-0.240274, -0.120346, 0.963216,
//0.000000, 0.992285, 0.123978;












//			double fov = 20;
//			double near = 0.5;
//			double far = 15.0;
//			double aspect = 1.333;


//	// HD RGB image with texture
//	string image_ref_path = "../data/rgb/HDdataset/rt_16_40_56_cam11__rgb.exr";
//	string image_target_path = "../data/rgb/HDdataset/rt_16_40_56_cam55__rgb.exr";

// HD BaseColor Image with texture
//	string image_ref_baseColor_path = "../data/rgb/HDdataset/rt_16_35_53_cam11__basecolor.exr";
//	string image_target_baseColor = "../data/rgb/HDdataset/rt_16_35_53_cam55__basecolor.exr";

// HD Depth map
//	string depth_ref_path = "../data/rgb/HDdataset/rt_16_36_54_cam11_depth.exr";
//	string depth_target_path = "../data/rgb/HDdataset/rt_16_36_54_cam55_depth.exr";
// Metallic and Roughness
//	string image_ref_MR_path = "../data/rgb/HDdataset/rt_16_47_3_cam11__mr.exr"; // store value in rgb channels,  channel b: metallic, channel green: roughness
//	string image_target_MR_path = "../data/rgb/HDdataset/rt_16_47_3_cam55__mr.exr";


//	vector<Mat>ch;
//	ch.push_back(grayImage_ref);
//	ch.push_back(grayImage_ref);
//	ch.push_back(grayImage_ref);
//	Mat trible_gray_image;
//
//	merge(ch,trible_gray_image );
//	imshow("trible_gray_image",trible_gray_image);
//
////	imshow("grayImage_target",grayImage_target);
//	waitKey(0);


//	Eigen::Matrix3d R1;
//	R1<< 1,0,0,
//		0,1,0,
//		0,0,1;
//	Eigen::Vector3d t1;
//	t1<< 1,2,3;
//	Sophus::SE3d Tran(R1,t1);
//	double *transl=Tran.data();
//	cout<<"show sophus data:"<<*(transl)<<","<<*(transl+1)<<","<<*(transl+2)<<","<<*(transl+3)<<","<<*(transl+4)<<","<<*(transl+5)<<","<<*(transl+6)<<"!"<<endl;
//	Eigen::Quaternion<double> q_new(*(transl+3),*(transl),*(transl+1),*(transl+2));
//	Eigen::Matrix<double, 3,1> translation(*(transl+4),*(transl+5),*(transl+6));
//	Eigen::Matrix<double,3,3> R_new;
//	R_new=q_new.normalized().toRotationMatrix();
//	cout<<"\n show rotation matrix:"<< R_new<<endl;
//	cout<<"\n show translation"<<translation<<endl;

// read target image
//	Mat image_target = imread(image_target_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
// color space conversion
//	cvtColor(image_target, grayImage_target, COLOR_BGR2GRAY);   right
//    Eigen::Vector2i pixel(213,295);
//	imageInfo(image_target_path,pixel);


// precision improvement
//	grayImage_target.convertTo(grayImage_target, CV_64FC1, 1.0 / 255.0);

// read ref image
//	Mat image_ref = imread(image_ref_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
// color space conversion
//	cvtColor(image_ref, grayImage_ref, COLOR_BGR2GRAY);   // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11 left

// precision improvement
//	grayImage_ref.convertTo(grayImage_ref, CV_64FC1, 1.0 / 255.0);

//	imageInfo(depth_ref_path,pixel_pos);


//	depth_ref.convertTo(depth_ref_render, CV_64FC1);
//	depth_ref= depth_ref_render *(60.0-0.01) + 0.01;

//		cv::minMaxIdx(depth_ref, &min, &max);
//		cout<<"\n show the depth_ref value range:\n"<<"min:"<<min<<"max:"<<max<<endl;
//		cout<<"depth of depth_ref"<<depth_ref.depth()<<"!!!!!!!!!!!!!!"<<endl;
//   cv::minMaxIdx(depth_ref, &min, &max);
//   cout<<"\n show the depth_ref value range:\n"<<"min:"<<min<<"max:"<<max<<endl;


//   depth_target.convertTo(depth_target, CV_64F);
//   depth_target = depth_target / 5000.0;

//	depth_target.convertTo(depth_tar_render,CV_64FC1);
//	depth_target=depth_tar_render *(60.0-0.01) + 0.01;

//			double fov_y= 33.398;
//			double near= 0.01;
//			double far= 60.0;
//			double aspect= 1.333;
//
//			M_matrix << 1.0/(tan(0.5*fov_y)*aspect), 0, 0, 0,
//					0,  atan(0.5*fov_y), 0   ,  0,
//					0,0, (far+near)/(near-far), 2*far*near/(near-far),
//					0,  0,   -1,    0;