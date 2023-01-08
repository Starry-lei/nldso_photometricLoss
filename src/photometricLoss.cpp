
#include <sophus/se3.hpp>
#include <unordered_map>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>


#include "preFilter/preFilter.h"
#include "dataLoader/dataLoader.h"
#include "deltaCompute/deltaCompute.h"
#include "utils/ultils.h"
#include "cameraProjection/reprojection.h"
#include "cameraProjection/photometricBA.h"

#include <algorithm>
using namespace cv;
using namespace std;
using namespace DSONL;


void GetFileNames(string path,vector<string>& filenames)
{
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str()))){
        cout<<"Folder doesn't Exist!"<<endl;
        return;
    }
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
            filenames.push_back(path + "/" + ptr->d_name);
        }
    }
    closedir(pDir);
}



int main(int argc, char **argv) {

    // TODO(Binghui): A and B to be parallelized

    // A: line: 26-27
    dataLoader *dataLoader = new DSONL::dataLoader();
	dataLoader->Init();


// load env light maps

//    std::string envMap_Folder="/media/lei/Data/Simulation Dataset/newpanorama";
//
//    std::vector<string> fileNames;
//    GetFileNames(envMap_Folder,fileNames);
//    cout<<" \n Show fileNames.size():"<<fileNames.size()<< endl;


    //B: line: 34- 45
    string  parameter_path = "include/EnvLight_Data/envMap01/parameters.csv";
    EnvMapLookup *EnvMapLookup=new DSONL::EnvMapLookup(argc,argv, parameter_path);
    EnvMapLookup->makeMipMap();
    delete EnvMapLookup;

    string parameters_path = "include/EnvLight_Data/envMap01/parameters_env_diffuse.csv";
    diffuseMap *diffuseMap = new DSONL::diffuseMap;
    diffuseMap->Init(argc,argv, parameters_path);
    diffuseMap->makeDiffuseMap();
    delete diffuseMap;

    string brdfIntegrationMap_path = "../include/brdfIntegrationMap/brdfIntegrationMap.pfm";
    brdfIntegrationMap *brdfIntegrationMap= new DSONL::brdfIntegrationMap(brdfIntegrationMap_path);
    brdfIntegrationMap->makebrdfIntegrationMap();
    delete brdfIntegrationMap;













    Mat image_ref_metallic = dataLoader->image_ref_metallic;
    Mat image_ref_roughness = dataLoader->image_ref_roughness;

	Mat grayImage_target, grayImage_ref, depth_ref, depth_target, image_ref_baseColor, image_target_baseColor;
	grayImage_ref = dataLoader->grayImage_ref;
	grayImage_target = dataLoader->grayImage_target;
	grayImage_ref.convertTo(grayImage_ref, CV_64FC1);
	grayImage_target.convertTo(grayImage_target, CV_64FC1);
	depth_ref = dataLoader->depth_map_ref;
	Mat depth_ref_GT = dataLoader->depth_map_ref;
	depth_target = dataLoader->depth_map_target;
	image_ref_baseColor = dataLoader->image_ref_baseColor;
	//        imshow("image_ref_baseColor",image_ref_baseColor);
	//        waitKey(0);
	image_target_baseColor = dataLoader->image_target_baseColor;
    Mat normal_map_GT;
    normal_map_GT = dataLoader->normal_map_GT;


	// show the depth image with noise
	double min_depth_val, max_depth_val;
	cv::minMaxLoc(depth_ref, &min_depth_val, &max_depth_val);
	cout << "\n show original depth_ref min, max:\n" << min_depth_val << "," << max_depth_val << endl;

	// grayImage_ref
    double min_radiance_val, max_radiance_val;
    cv::minMaxLoc(grayImage_ref, &min_radiance_val, &max_radiance_val);
    cout << "\n show original grayImage_ref min, max:\n" << min_radiance_val << "," << max_radiance_val << endl;

	Eigen::Matrix3f K;
	K = dataLoader->camera_intrinsics;

	imshow("grayImage_ref",grayImage_ref);
	imshow("grayImage_target",grayImage_target);
	waitKey(0);


	// ----------------------------------------optimization variable: R, t--------------------------------------
	Sophus::SE3d xi, xi_GT;
	Eigen::Matrix3d Camera1_c2w= dataLoader->R1;
	Eigen::Matrix<double,3,3> R;
	R = dataLoader->q_12.normalized().toRotationMatrix();
	xi_GT.setRotationMatrix(R);
	xi_GT.translation() = dataLoader->t12;


	// ----------------------------------------optimization variable: depth --------------------------------------
	cout << "\n Show GT rotation:\n" << xi_GT.rotationMatrix() << "\n Show GT translation:\n" << xi_GT.translation()
	     << endl;
	// -------------------------------------------------Movingleast algorithm-----------------------------------------
	std::vector<Eigen::Vector3d> pts;
	cv::Mat normal_map(depth_ref.rows, depth_ref.cols, CV_32FC3);
	//MLS();
	//--------------------------------------------------normal_map_GT---------------------------------------------------


	for (int u = 0; u < depth_ref.rows; u++) // colId, cols: 0 to 480
	{
		for (int v = 0; v < depth_ref.cols; v++) // rowId,  rows: 0 to 640
		{

			if (depth_ref.at<float>(u,v)==15.0f){ continue; }
			Eigen::Vector3f normal_new(normal_map_GT.at<cv::Vec3f>(u, v)[2], normal_map_GT.at<cv::Vec3f>(u, v)[1],normal_map_GT.at<cv::Vec3f>(u, v)[0]);
			normal_new = (dataLoader->R1.cast<float>()).transpose()* normal_new;
			Eigen::Vector3f principal_axis(0, 0, 1);
			if (normal_new.dot(principal_axis) > 0) {normal_new = -normal_new;}
			normal_map.at<Vec3f>(u, v)[0] = normal_new(0);
			normal_map.at<Vec3f>(u, v)[1] = normal_new(1);
			normal_map.at<Vec3f>(u, v)[2] = normal_new(2);

		}
	}

//	//-------------------------------------------------Data perturbation--------------------------------------------------------------------
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
	//	float densities[] = {0.03, 0.003, 0.05, 0.15, 0.5, 1}; /// number of optimized depths,  current index is 1


	PhotometricBAOptions options;
	Mat newNormalMap = normal_map;
	double distanceThres = 0.007;
	float upper = 9.0;
	float buttom = 0.002;
	float up_new = upper;
	float butt_new = buttom;
	Mat deltaMap(depth_ref.rows, depth_ref.cols, CV_32FC1, Scalar(1)); // storing delta
	int lvl_target, lvl_ref;

	double depth_upper_bound = 0.5;  // 0.5; 1
	double depth_lower_bound = 0.1;  // 0.001

	options.optimize_depth = false;
	options.useFilterController = false; // control the number of optimized depth
	options.optimize_pose = true;
	options.use_huber = false;
	options.lambertianCase = false;
	options.usePixelSelector = false;
	dataLoader->options_.remove_outlier_manually = false;
	options.huber_parameter = 0.25*4.0 / 255.0;   /// 0.25*4/255 :   or 4/255

	// -----------------------------------------Initialize the pose xi with GT or just use the default value---------------------
//	Eigen::Vector3d initial_translation;
//	initial_translation<< -0.2, -0.1, 0;

	xi.setRotationMatrix(perturbedRotation);
	xi.translation() = perturbedTranslation;

//	xi.translation() = initial_translation;

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
	cout << "\n Show initial rotation:\n" << Rotation.matrix() << "\n Show initial translation:\n" << Translation<< endl;
	cout << "\nShow current rotation perturbation error :" << roErr<< "\n Show current translation perturbation error : " << trErr << "\nShow current depth perturbation error :"<< depth_Error << endl;

	double min_gt_special, max_gt_special;
	cv::minMaxLoc(inv_depth_ref, &min_gt_special, &max_gt_special);
	cout << "\n show inv_depth_ref min, max:\n" << min_gt_special << "," << max_gt_special << endl;

	//	Mat inv_depth_ref_for_show = inv_depth_ref * (1.0 / (max_gt_special - min_gt_special)) +(-min_gt_special * (1.0 / (max_gt_special - min_gt_special)));
	//	string depth_ref_name = "inv_depth_ref";
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
		while (i < 2) {
			double max_n_, min_n_;
			cv::minMaxLoc(deltaMap, &min_n_, &max_n_);
			cout << "->>>>>>>>>>>>>>>>>show max and min of estimated deltaMap:" << max_n_ << "," << min_n_ << endl;
			Mat mask = cv::Mat(deltaMap != deltaMap);
			deltaMap.setTo(1.0, mask);
			if (i == 1) {
				cout << "depthErr(depth_ref_gt, inv_depth_ref).val[0]:" << depthErr(depth_ref_gt, inv_depth_ref).val[0]<< endl;
				showScaledImage(depth_ref_NS_before, depth_ref_gt, inv_depth_ref);
			}
			cv::minMaxLoc(inv_depth_ref, &min_gt_special, &max_gt_special);
			cout << "\n show inv_depth_ref min, max:\n" << min_gt_special << "," << max_gt_special << endl;
			Mat inv_depth_ref_for_show = inv_depth_ref * (1.0 / (max_gt_special - min_gt_special)) +(-min_gt_special * (1.0 / (max_gt_special - min_gt_special)));
			string depth_ref_name = "inv_depth_ref" + to_string(i);
//			imshow(depth_ref_name, inv_depth_ref_for_show);
//			cout<<"show the current depth:"<<inv_depth_ref.at<double>(359,470)<<endl;


			if (dataLoader->options_.remove_outlier_manually) {
				//				PhotometricBA(IRef, I, options, Klvl, xi, inv_depth_ref,deltaMap,depth_upper_bound, depth_lower_bound,dataLoader->outlier_mask_big_baseline);
				//				//				inv_depth_ref.convertTo(inv_depth_ref, CV_32FC1);
				//				//				showScaledImage(depth_ref_gt, inv_depth_ref);
				//				//				waitKey(0);
			} else {
				PhotometricBA(IRef, I, options, Klvl, Rotation, Translation, inv_depth_ref, deltaMap, depth_upper_bound,depth_lower_bound, statusMap, statusMapB);
			}

			DSONL::updateDelta(Camera1_c2w,Rotation,Translation,Klvl,image_ref_baseColor,inv_depth_ref,image_ref_metallic ,image_ref_roughness,deltaMap,newNormalMap,up_new, butt_new);

//			Mat deltaMapGT_res= deltaMapGT(grayImage_ref,depth_ref,grayImage_target,depth_target,K.cast<double>(),distanceThres,xi_GT, upper, buttom, deltaMap);
//			Mat showGTdeltaMap=colorMap(deltaMapGT_res, upper, buttom);
//			Mat showESdeltaMap=colorMap(deltaMap, upper, buttom);
//			imshow("show GT deltaMap", showGTdeltaMap);
//			imshow("show ES deltaMap", showESdeltaMap);


//			/// TEMP TEST BEGIN
//			for (int x = 0; x < deltaMapGT_res.rows; ++x) {
//				for (int y = 0; y < deltaMapGT_res.cols; ++y) {
//					if (deltaMapGT_res.at<float>(x, y) == -1) {deltaMap.at<float>(x, y) =1;}
//				}
//			}
//			double max_n_1, min_n_1;
//			cv::minMaxLoc(deltaMapGT_res, &min_n_1, &max_n_1);
//			cout << "->>>>>>>>>>>>>>>>>show max and min of deltaMapGT_res<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<:" << max_n_1 << "," << min_n_1 << endl;
			//->>>>>>>>>>>>>>>>>show max and min of deltaMapGT_res<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<:8.7931,0.141176
//			 deltaMap=deltaMapGT_res.clone();// !!!!!!!!!!!!!!!!!!!!!!!!!!!!test!!!!!!!!!!!!!!!!!!!!!!!!!!!11
//            /// TEMP TEST END

//			imwrite("GT_deltaMap.exr", showGTdeltaMap);
//			imwrite("ES_deltaMap.exr", showESdeltaMap);

			cout << "\n show depth_ref min, max:\n" << min_gt_special << "," << max_gt_special << endl;
			cout << "\n Show optimized rotation:\n" << Rotation.matrix() << "\n Show optimized translation:\n"<< Translation << endl;
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
//		waitKey(0);

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








// case 1
// GLI 0.5, 0.5

//	============SampleBrdf val(RGBA):
//	1,0.837255,0.393137,1
// GSN:
//	image_ref_path_PFM blue,green and red channel value: [0.394053, 0.837957, 1.81198]


// case 2

// GSN 0.5, 0.75
//	image_ref_path_PFM blue,green and red channel value:
//	[0.378213, 0.753771, 1.5108]

// GLI : 0.5, 0.25(different order)
//============SampleDiffuse val(RGBA):
//    1.51918,0.760294,0.382353,1
// same order
//============SampleDiffuse val(RGBA):
//    0.382353,0.760294,1.51918,1



//	imshow("show Diffuse Map", getDiffuseMap.diffuse_Map);

//	brdfIntegrationMap *brdfIntegrationMap;
//	brdfIntegrationMap= new DSONL::brdfIntegrationMap;
//	float NoV= 0.5f, roughness= 0.25f;
//	gli::vec4 test_brdf= brdfIntegrationMap->get_brdfIntegrationMap(NoV, roughness);
//

//	for (int i = 0; i < 6; ++i) {
//		imshow("image"+ to_string(i), img_pyramid_mask[i]);
//
//		// show the min max val
//		double min_depth_val, max_depth_val;
//		cv::minMaxLoc( EnvMapLookup.image_pyramid[i], &min_depth_val, &max_depth_val);
//		cout<<"\n show  EnvMapLookup.image_pyramid[i] min, max:\n"<<min_depth_val<<","<<max_depth_val<<endl;
//	}
//	waitKey(0);


//        DSONL::IBL_Radiance radiance_beta_vec;
//        Vec3f testRadianceVal=  radiance_beta_vec.specularIBL(Vec3f(0.5,0.5,0.5), 0.5, Vec3f(0.5,0.75,0.25), Vec3f(0.5,0.5,0.5) );
//        cout<<"=======show testRadianceVal:"<<testRadianceVal<<endl;
// TEST specular:

// GSN: for comparision: image_ref_path_PFM blue,green and red channel value: [0.0191865, 0.0430412, 0.0789493]
// GLI:=======show testRadianceVal:[0.022193, 0.043398, 0.0780335] // passed


// GLI without 0.5* =======show testRadianceVal:[0.0228224, 0.0450064, 0.0824027]
// GLI: normalize V vector: ======show testRadianceVal:[0.0230061, 0.0451581, 0.0816396]
// GLI:==============show testRadianceVal:[0.013847, 0.0262424, 0.0450112]
// image_ref_path_PFM blue:  green and red channel value: [0.0130103, 0.0246642, 0.0424151]
//        Vec3f normal=Vec3f(0.8, 0.6, 0.5);
//        Vec3f testRadianceVal_testdiffuse=  radiance_beta_vec.diffuseIBL(normalize(normal));
//        cout<<"=======show testRadianceVal_testdiffuse:"<<testRadianceVal_testdiffuse<<endl;
//        // TEST diffuse:
//        // GLI: diffuse: =======show testRadianceVal_testdiffuse:[0.0347963, 0.067812, 0.123859]????
//        // GSN: image_ref_path_PFM blue,green and red channel value: [0.0339334, 0.0683846, 0.120991]
//        Vec3f testRadianceVal_testfreshSchlick=  radiance_beta_vec.fresnelSchlick(0.5, Vec3f(0.5,0.5,0.5));
//        cout<<"=======show testRadianceVal_testfreshSchlick:"<<testRadianceVal_testfreshSchlick<<endl;
