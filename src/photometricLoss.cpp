
#include <sophus/se3.hpp>
#include <unordered_map>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/eigen.hpp>

#include "preFilter/preFilter.h"
#include "dataLoader/dataLoader.h"
#include "deltaCompute/deltaCompute.h"
#include "utils/ultils.h"
#include "cameraProjection/reprojection.h"
#include "cameraProjection/photometricBA.h"


using namespace cv;
using namespace std;
using namespace DSONL;


int main(int argc, char **argv) {


	dataLoader *dataLoader = new DSONL::dataLoader();
	dataLoader->Init();



        EnvMapLookup *EnvMapLookup=new DSONL::EnvMapLookup(argc,argv);
        EnvMapLookup->makeMipMap(prefilteredEnvmapSampler);


        gli::vec4 Sample_val =DSONL:: prefilteredEnvmapSampler->texture_lod(gli::fsampler2D::normalized_type(0.5f, 0.75f),0.0f); // transform the texture coordinate
        cout << "\n============Sample_val val(RGBA):\n" << Sample_val.b << "," << Sample_val.g << "," << Sample_val.r << ","   << Sample_val.a << endl;
        delete DSONL::prefilteredEnvmapSampler;



        float image_ref_metallic = dataLoader->image_ref_metallic;
	float image_ref_roughness = dataLoader->image_ref_roughness;

	Mat grayImage_target, grayImage_ref, depth_ref, depth_target, image_ref_baseColor, image_target_baseColor;
	grayImage_ref = dataLoader->grayImage_ref;
	grayImage_target = dataLoader->grayImage_target;
	grayImage_ref.convertTo(grayImage_ref, CV_64FC1);
	grayImage_target.convertTo(grayImage_target, CV_64FC1);

	depth_ref = dataLoader->depth_map_ref;
	Mat depth_ref_GT = dataLoader->depth_map_ref;

	depth_target = dataLoader->depth_map_target;
	image_ref_baseColor = dataLoader->image_ref_baseColor;
	image_target_baseColor = dataLoader->image_target_baseColor;


	// show the depth image with noise
	double min_depth_val, max_depth_val;
	cv::minMaxLoc(depth_ref, &min_depth_val, &max_depth_val);
	cout << "\n show original depth_ref min, max:\n" << min_depth_val << "," << max_depth_val << endl;


	Eigen::Matrix3f K;
	K = dataLoader->camera_intrinsics;




	//	imshow("grayImage_ref",grayImage_ref);
	//	imshow("grayImage_target",grayImage_target);
	//	waitKey(0);


	// ----------------------------------------optimization variable: R, t--------------------------------------
	Sophus::SE3d xi, xi_GT;
	//	Sophus::SO3d Rotation;
	//	Eigen::Matrix<double, 3,1> Translation;

	Eigen::Matrix<double, 3, 3> R;
	R = dataLoader->q_12.normalized().toRotationMatrix();
	xi_GT.setRotationMatrix(R);
	xi_GT.translation() = dataLoader->t12;

	// ----------------------------------------optimization variable: depth --------------------------------------
	cout << "\n Show GT rotation:\n" << xi_GT.rotationMatrix() << "\n Show GT translation:\n" << xi_GT.translation()
	     << endl;

	// ----------------------------------------Movingleast algorithm---------------------------------------------------------------
	std::vector<Eigen::Vector3d> pts;
	cv::Mat normal_map(depth_ref.rows, depth_ref.cols, CV_64FC3);
	//MLS();
	//	---------------------------------------------------------normal_map_GT---------------------------------------------------
	Mat normal_map_GT;
	normal_map_GT = dataLoader->normal_map_GT;
	for (int u = 0; u < depth_ref.rows; u++) // colId, cols: 0 to 480
	{
		for (int v = 0; v < depth_ref.cols; v++) // rowId,  rows: 0 to 640
		{

			Eigen::Vector3d normal_new(normal_map_GT.at<cv::Vec3f>(u, v)[2], -normal_map_GT.at<cv::Vec3f>(u, v)[1],
			                           normal_map_GT.at<cv::Vec3f>(u, v)[0]);
			normal_new = dataLoader->R1.transpose() * normal_new;

			Eigen::Vector3d principal_axis(0, 0, 1);
			if (normal_new.dot(principal_axis) > 0) {
				normal_new = -normal_new;
			}

			normal_map.at<Vec3d>(u, v)[0] = normal_new(0);
			normal_map.at<Vec3d>(u, v)[1] = normal_new(1);
			normal_map.at<Vec3d>(u, v)[2] = normal_new(2);

		}
	}


//	//	--------------------------------------------------------------------Data perturbation--------------------------------------------------------------------
//	// Add noise to original depth image, depth_ref_NS
	Mat inv_depth_ref, depth_ref_gt;
	Mat depth_ref_NS;
	double roErr;
	Eigen::Matrix3d R_GT(xi_GT.rotationMatrix());
	Eigen::Matrix3d perturbedRotation = rotation_pertabation(0.0, 0.0, 0.0, R_GT, roErr); // degree

	double trErr;
	Eigen::Vector3d T_GT(xi_GT.translation());
	Eigen::Vector3d perturbedTranslation = translation_pertabation(0.0, 0.0, 0.0, T_GT, trErr); // percentage

	double Mean = 0.0, StdDev = 0;
	float densities[] = {0.03, 0.003, 0.05, 0.15, 0.5, 1}; /// number of optimized depths,  current index is 1


	PhotometricBAOptions options;
	Mat newNormalMap = normal_map;
	double distanceThres = 0.07;
	float upper = 5;
	float buttom = 0.2;
	float up_new = upper;
	float butt_new = buttom;
	Mat deltaMap(depth_ref.rows, depth_ref.cols, CV_32FC1, Scalar(1)); // storing delta
	int lvl_target, lvl_ref;

	double depth_upper_bound = 0.5;  // 0.5; 1
	double depth_lower_bound = 0.1;  // 0.001

	options.optimize_depth = false;
	options.useFilterController = false; // control the number of optimized depth
	options.optimize_pose = true;
	options.use_huber = true;
	options.lambertianCase = false;
	options.usePixelSelector = false;
	dataLoader->options_.remove_outlier_manually = false;
	options.huber_parameter = 0.25 * 4.0 / 255.0;   /// 0.25*4/255 :   or 4/255

	// initialize the pose xi         or just use the default value
//	xi.setRotationMatrix(perturbedRotation);
//	xi.translation() = perturbedTranslation;


	Sophus::SO3d Rotation(xi.rotationMatrix());
	Eigen::Matrix<double, 3, 1> Translation(xi.translation());


//	PixelSelector* pixelSelector=NULL;
//	FrameHessian* newFrame_ref=NULL;
//	FrameHessian* newFrame_tar=NULL;
//	FrameHessian* depthMap_ref=NULL;
	float *color_ref = NULL;
	float *color_tar = NULL;
	float *depthMapArray_ref = NULL;
	float *statusMap = NULL;
	bool *statusMapB = NULL;

	AddGaussianNoise_Opencv(depth_ref, depth_ref_NS, Mean, StdDev, statusMap);
	divide(Scalar(1), depth_ref, depth_ref_gt);
	divide(Scalar(1), depth_ref_NS, inv_depth_ref);
	Mat inv_depth_tar;
	divide(Scalar(1), depth_target, inv_depth_tar);


	Mat depth_ref_NS_before = inv_depth_ref.clone();
	double min_inv, max_inv;
	cv::minMaxLoc(inv_depth_ref, &min_inv, &max_inv);
	cout << "\n show original inv_depth_ref min, max:\n" << min_inv << "," << max_inv << endl;
	Scalar_<double> depth_Err = depthErr(depth_ref_gt, inv_depth_ref);
	double depth_Error = depth_Err.val[0];
	cout << "\n Show initial rotation:\n" << Rotation.matrix() << "\n Show initial translation:\n" << Translation
	     << endl;
	cout << "\nShow current rotation perturbation error :" << roErr
	     << "\n Show current translation perturbation error : " << trErr << "\nShow current depth perturbation error :"
	     << depth_Error << endl;

	double min_gt_special, max_gt_special;
	cv::minMaxLoc(inv_depth_ref, &min_gt_special, &max_gt_special);
	cout << "\n show inv_depth_ref min, max:\n" << min_gt_special << "," << max_gt_special << endl;
	Mat inv_depth_ref_for_show = inv_depth_ref * (1.0 / (max_gt_special - min_gt_special)) +
	                             (-min_gt_special * (1.0 / (max_gt_special - min_gt_special)));
	string depth_ref_name = "inv_depth_ref";
//	imshow(depth_ref_name, inv_depth_ref_for_show);


	for (int lvl = 1; lvl >= 1; lvl--) {

		cout << "\n Show the value of lvl:" << lvl << endl;
		Mat IRef, DRef, I, D;
		Eigen::Matrix3f Klvl, Klvl_ignore;
		lvl_target = lvl;
		lvl_ref = lvl;

		downscale(grayImage_ref, inv_depth_ref, K, lvl_ref, IRef, DRef, Klvl);
		downscale(grayImage_target, depth_target, K, lvl_target, I, D, Klvl_ignore);
		double min_gt_special, max_gt_special;

		int i = 0;
		while (i < 1) {
			double max_n_, min_n_;
			cv::minMaxLoc(deltaMap, &min_n_, &max_n_);
			cout << "->>>>>>>>>>>>>>>>>show max and min of estimated deltaMap:" << max_n_ << "," << min_n_ << endl;
			Mat mask = cv::Mat(deltaMap != deltaMap);
			deltaMap.setTo(1.0, mask);
			if (i == 1) {
				cout << "depthErr(depth_ref_gt, inv_depth_ref).val[0]:" << depthErr(depth_ref_gt, inv_depth_ref).val[0]
				     << endl;
				showScaledImage(depth_ref_NS_before, depth_ref_gt, inv_depth_ref);
			}
			cv::minMaxLoc(inv_depth_ref, &min_gt_special, &max_gt_special);
			cout << "\n show inv_depth_ref min, max:\n" << min_gt_special << "," << max_gt_special << endl;
			Mat inv_depth_ref_for_show = inv_depth_ref * (1.0 / (max_gt_special - min_gt_special)) +
			                             (-min_gt_special * (1.0 / (max_gt_special - min_gt_special)));
			string depth_ref_name = "inv_depth_ref" + to_string(i);
//			imshow(depth_ref_name, inv_depth_ref_for_show);
//			cout<<"show the current depth:"<<inv_depth_ref.at<double>(359,470)<<endl;


			if (dataLoader->options_.remove_outlier_manually) {
//				PhotometricBA(IRef, I, options, Klvl, xi, inv_depth_ref,deltaMap,depth_upper_bound, depth_lower_bound,dataLoader->outlier_mask_big_baseline);
//				//				inv_depth_ref.convertTo(inv_depth_ref, CV_32FC1);
//				//				imwrite("test_inv_depth.exr",inv_depth_ref);
//				//				showScaledImage(depth_ref_gt, inv_depth_ref);
//				//				waitKey(0);
			} else {
//				PhotometricBA(IRef, I, options, Klvl, xi, inv_depth_ref,deltaMap,depth_upper_bound, depth_lower_bound, statusMap, statusMapB);
				PhotometricBA(IRef, I, options, Klvl, Rotation, Translation, inv_depth_ref, deltaMap, depth_upper_bound,
				              depth_lower_bound, statusMap, statusMapB);

//				imshow(depth_ref_name, inv_depth_ref_for_show);
//				waitKey(0);
			}


//			updateDelta(xi,Klvl,image_ref_baseColor,depth_ref,image_ref_metallic ,image_ref_roughness,light_source, deltaMap,newNormalMap,up_new, butt_new);


//			Mat deltaMapGT_res= deltaMapGT(grayImage_ref,depth_ref,grayImage_target,depth_target,K,distanceThres,xi_GT, upper, buttom, deltaMap);
//			Mat showGTdeltaMap=colorMap(deltaMapGT_res, upper, buttom);
//			Mat showESdeltaMap=colorMap(deltaMap, upper, buttom);
//			imshow("show GT deltaMap", showGTdeltaMap);
//			imshow("show ES deltaMap", showESdeltaMap);
//			imwrite("GT_deltaMap.exr",showGTdeltaMap);
//			imwrite("ES_deltaMap.exr",showESdeltaMap);
//
			cout << "\n show depth_ref min, max:\n" << min_gt_special << "," << max_gt_special << endl;
			cout << "\n Show optimized rotation:\n" << Rotation.matrix() << "\n Show optimized translation:\n"
			     << Translation << endl;
			cout << "\n Show Rotational error :" << rotationErr(xi_GT.rotationMatrix(), Rotation.matrix())
			     << "(degree)." << "\n Show translational error :"
			     << 100 * translationErr(xi_GT.translation(), Translation) << "(%) "
			     << "\n Show depth error :" << depthErr(depth_ref_gt, inv_depth_ref).val[0]
			     << endl;// !!!!!!!!!!!!!!!!!!!!!!!!

			i += 1;

		}
		cout << "\nShow current rotation perturbation error :" << roErr
		     << "\nShow current translation perturbation error : " << trErr
		     << "\nShow current depth perturbation error :" << depth_Error << endl;
		waitKey(0);

	}

	// tidy up
	delete dataLoader;
	if (options.usePixelSelector) {
//		delete pixelSelector;
//		delete newFrame_ref;
//		delete newFrame_tar;
//		delete depthMap_ref;
		delete[] statusMap;
		delete[] color_ref;
		delete[] color_tar;
		delete[] depthMapArray_ref;
		delete[] statusMap;
		delete[] statusMapB;
	}
	return 0;
}




//	float u= 0.5f, v= 0.25f;
//	diffuseMapMask DiffuseMaskMap;
//	DiffuseMaskMap.Init(argc,argv);
//	DiffuseMaskMap.getDiffuseMask(u,v);
//
//
//
//
//
//	diffuseMap getDiffuseMap;
//	getDiffuseMap.Init(argc,argv);
//	getDiffuseMap.getDiffuse(u, v);
//
//
//	Mat final_diffuseMap;
//	makeDiffuseMap(getDiffuseMap.diffuse_Map, DiffuseMaskMap.diffuse_Map_Mask, final_diffuseMap);
//
//	getDiffuseMap.diffuse_Map=final_diffuseMap;
//	cout<<"show final search value:"<<endl;
//	getDiffuseMap.getDiffuse(u, v);
//
//



// case 1
// GLI 0.5, 0.5

//	============SampleBrdf val(RGBA):
//	1,0.837255,0.393137,1
// GSN:
//	image_ref_path_PFM blue  green and red channel value: [0.394053, 0.837957, 1.81198]


// case 2

// GSN 0.5, 0.75
//	image_ref_path_PFM blue  green and red channel value:
//	[0.378213, 0.753771, 1.5108]

// GLI : 0.5, 0.25
//	============SampleDiffuse val(RGBA):
//	1,0.760294,0.382353,1

//	imshow("show Diffuse Map", getDiffuseMap.diffuse_Map);

//	brdfIntegrationMap *brdfIntegrationMap;
//	brdfIntegrationMap= new DSONL::brdfIntegrationMap;
//	float NoV= 0.5f, roughness= 0.25f;
//	gli::vec4 test_brdf= brdfIntegrationMap->get_brdfIntegrationMap(NoV, roughness);
//
//
//
//
//	EnvMapLookup EnvMapLookup(argc,argv);
//	EnvMapLookup.makeMipMap();
//


//	for (int i = 0; i < 6; ++i) {
//		imshow("image"+ to_string(i), EnvMapLookup.image_pyramid[i]);
//
////		// show the min max val
////		double min_depth_val, max_depth_val;
////		cv::minMaxLoc( EnvMapLookup.image_pyramid[i], &min_depth_val, &max_depth_val);
////		cout<<"\n show  EnvMapLookup.image_pyramid[i] min, max:\n"<<min_depth_val<<","<<max_depth_val<<endl;
//	}
//	waitKey(0);


