
#include <algorithm>
#include <sophus/se3.hpp>
#include <unordered_map>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>

#include "dataLoader/dataLoader.h"
#include "utils/ultils.h"
#include "cameraProjection/reprojection.h"
#include "cameraProjection/photometricBA.h"
#include "controlPointSelector/ctrlPointSelector.h"
#include "deltaCompute/deltaCompute.h"
#include "pixelSelector.h"
#include "SpecularHighlightRemoval/SpecularHighlightRemoval.h"
using namespace cv;
using namespace std;
using namespace DSONL;

// control points  pose # qw qx qy qz x y z
int main(int argc, char **argv) {

    // A: line: 26-27
    dataLoader *dataLoader = new DSONL::dataLoader();
	dataLoader->Init();


    // ===========================Environment Light preprocessing module===========================================

//    std::string envMap_Folder="/home/lei/Documents/Research/envMapData/EnvMap150_wholeImg";
//    string controlPointPose_path= "/home/lei/Documents/Research/envMapData/2frame0370_02_control_cam_pose_150.txt";
//    string  renderedEnvMapPath=   "/home/lei/Documents/Research/envMapData/EnvMap150_wholeImg";

    std::string envMap_Folder=    "/home/lei/Documents/Research/envMapData/EnvMap_156ctrlPoints";
    string controlPointPose_path= "/home/lei/Documents/Research/envMapData/2frame0370_02_control_cam_pose156.txt";
    string  renderedEnvMapPath=   "/home/lei/Documents/Research/envMapData/EnvMap_156ctrlPoints";

    // data loading
    Mat Image_tar8UC3, Image_ref8UC3, depth_ref,depth_ref_inv, depth_tar_inv,depth_target;
    Mat normal_map_GT,image_ref_roughness,depth_ref_GT;
    Eigen::Matrix3f K_synthetic;

    normal_map_GT = dataLoader->normal_map_GT;




    image_ref_roughness = dataLoader->image_ref_roughness;
    Image_ref8UC3 = dataLoader->grayImage_ref;
    Image_tar8UC3 = dataLoader->grayImage_target;
	depth_ref_inv = dataLoader->depth_map_ref;
    depth_tar_inv = dataLoader->depth_map_target;
    K_synthetic = dataLoader->camera_intrinsics;
    divide(Scalar(1), depth_ref_inv, depth_ref_GT);
    float roughness_threshold = 0.3;



    // ===========================ctrlPoint Selector==========================================
    ctrlPointSelector  * ctrlPoint_Selector= new ctrlPointSelector(dataLoader->camPose1, controlPointPose_path,Image_ref8UC3, depth_ref_GT,K_synthetic);
    envLightLookup  *EnvLightLookup= new envLightLookup(ctrlPoint_Selector->selectedIndex, argc, argv, envMap_Folder,controlPointPose_path);

    cout<<"\n The preComputation of EnvMap is ready!"<<endl;
    // =============================intensity segmentation module===========================================
    Mat specular_diffuse_transition_mask(Image_ref8UC3.rows, Image_ref8UC3.cols,CV_8UC1,Scalar(0));
    Mat diffuse_mask(Image_ref8UC3.rows, Image_ref8UC3.cols,CV_8UC1,Scalar(0));

    SpecularHighlightRemoval specularHighlightRemoval;
    SpecularHighlightRemoval specularHighlightRemoval_tar;
    specularHighlightRemoval.initialize(Image_ref8UC3.rows, Image_ref8UC3.cols);
    specularHighlightRemoval_tar.initialize(Image_ref8UC3.rows, Image_ref8UC3.cols);

    Mat diffuseImage = specularHighlightRemoval.run(Image_ref8UC3);
    Mat diffuseImage_tar = specularHighlightRemoval_tar.run(Image_tar8UC3);

    Mat specularImage= specularHighlightRemoval.specularImage;
    Mat Img_range=specularHighlightRemoval.rangeImage;
    Mat clusterImage=specularHighlightRemoval.clusterImage; //  1: specular_diffuse transition, 2: diffuse, 3: specular


    // select a single channel of image
    int channelIdx = 1;// green channel
    Mat grayImage_ref,grayImage_ref_pS, grayImage_tar,grayImage_tar_pS, grayImage_ref_green, grayImage_tar_green;

//    extractChannel(Image_ref8UC3, grayImage_ref_green, channelIdx);
//    cvtColor(Image_ref8UC3, grayImage_ref, CV_BGR2GRAY);

    extractChannel(diffuseImage, grayImage_ref_green, channelIdx);
    extractChannel(diffuseImage_tar, grayImage_tar_green, channelIdx);

    cvtColor(diffuseImage, grayImage_ref_pS, CV_BGR2GRAY);
    cvtColor(diffuseImage_tar, grayImage_tar_pS, CV_BGR2GRAY);
    cvtColor(Image_ref8UC3, grayImage_ref, CV_BGR2GRAY);
    cvtColor(Image_tar8UC3, grayImage_tar, CV_BGR2GRAY);

    Mat grayImg, mat_mean, mat_stddev;
    double mean_val;
    double std_dev;
    meanStdDev(grayImage_ref_pS, mat_mean, mat_stddev);
    mean_val= mat_mean.at<double>(0,0);
    std_dev = mat_stddev.at<double>(0,0);


    // ====================================== pointSelector========================================
    bool usePixelSelector= true;
    float densities[] = {1,0.5,0.15,0.05,0.03}; // 不同层取得点密度
//    float densities[] = {0.03,0.003, 0.05,0.15,0.5,1}; /// number of optimized depths,  current index is 1
    PixelSelector* pixelSelector=NULL;
    FrameHessian* newFrame_ref=NULL;
    FrameHessian* newFrame_tar=NULL;
    FrameHessian* depthMap_ref=NULL;
    float* color_ref=NULL;
    float* color_tar=NULL;
    float* depthMapArray_ref=NULL;
    float* statusMap=NULL;
    bool*  statusMapB=NULL;
    Mat transitionField(grayImage_ref.rows,  grayImage_ref.cols, CV_8UC1, Scalar(0));

    bool preusePixelSelector= false;
    if (preusePixelSelector){
        double min_gray,max_gray;
        imshow("grayImage_ref",grayImage_ref_pS);
        newFrame_ref= new FrameHessian();
        newFrame_tar= new FrameHessian();
        pixelSelector= new PixelSelector(wG[0],hG[0]);
        color_ref= new float[wG[0]*hG[0]];
        color_tar= new float[wG[0]*hG[0]];
        for (int row = 0; row < hG[0]; ++row) {
            uchar *pixel_ref=grayImage_ref_pS.ptr<uchar>(row);
            uchar *pixel_tar=grayImage_tar_pS.ptr<uchar>(row);
            for (int col = 0; col < wG[0]; ++col) {
                color_ref[row*wG[0]+col]= (float) pixel_ref[col];
                color_tar[row*wG[0]+col]= (float)pixel_tar[col];
//              depthMapArray_ref[row*wG[0]+col]=pixel_depth_ref[col];
            }
        }
        newFrame_ref->makeImages(color_ref); // make image_ref pyramid
        newFrame_tar->makeImages(color_tar); // make image_tar pyramid


        statusMap= new float[wG[0]*hG[0]];
        statusMapB = new bool[wG[0]*hG[0]];
        int setting_desiredImmatureDensity=1500;
        float densities[] = {1,0.5,0.15,0.05,0.03}; // 不同层取得点密度
        int  npts[pyrLevelsUsed];
        Mat selectedPointMask1(grayImage_ref.rows,  grayImage_ref.cols, CV_8UC1, Scalar(0));
        pixelSelector->currentPotential= 3;
        for(int i=0; i>=0; i--) {
            cout << "\n pyrLevelsUsed:" << i << endl;
//			plotImPyr(newFrame_ref, i, "newFrame_ref");
//			plotImPyr(newFrame_tar, i, "newFrame_tar");
//			plotImPyr(depthMap_ref, i, "depthMap_ref");
            npts[i] = pixelSelector->makeMaps(newFrame_ref, statusMap, densities[1] * wG[0] * hG[0], 1, false, 2);
            cout << "\n npts[i]: " << npts[i] << "\n densities[i]*wG[0]*hG[0]:" << densities[i] * wG[0] * hG[0] << endl;
//            waitKey(0);
//            cv::Mat image_ref(hG[i], wG[i], CV_32FC1);
//            memcpy(image_ref.data, newFrame_ref->img_pyr[i], wG[i] * hG[i] * sizeof(float));
//            cv::Mat image_tar(hG[i], wG[i], CV_32FC1);
//            memcpy(image_tar.data, newFrame_tar->img_pyr[i], wG[i] * hG[i] * sizeof(float));
        }


//        float scale_std= 0.6; // LDR
//        float scale_std= 0.6; // HDR maybe wrong
        float scale_std= 0.8; // HDR only for test



        int point_counter=0;
        int dso_point_counter=0;
        for (int u = 0; u< grayImage_ref.rows; u++) // colId, cols: 0 to 480
        {
            for (int v = 0; v < grayImage_ref.cols; v++) // rowId,  rows: 0 to 640
            {
                // || clusterImage.at<int>(u, v) == 1
                if ((statusMap!=NULL && statusMap[u*grayImage_ref.cols+v]!=0) ){
                    dso_point_counter+=1;
                    selectedPointMask1.at<uchar>(u,v)= 255;
                    // ================================save the selectedPoint mask here=======================
                    // apply the intensity segmentation here
                    // && (grayImage_ref.at<uchar>(u,v)>(mean_val+scale_std*std_dev))
                    if ( (image_ref_roughness.at<float>(u,v) < roughness_threshold)  && clusterImage.at<int>(u, v) == 1 ){
                        transitionField.at<uchar>(u,v)= 255;
                        statusMap[u*grayImage_ref.cols+v]=255;
                        point_counter+=1;
                    }

                }
            }
        }
        // refine the point selector
        Mat inputImage= Image_ref8UC3.clone();
        Mat inputImage_copy_1= inputImage.clone();
        Mat inputImage_copy_2= inputImage.clone();
        Mat inputImage_copy_3= inputImage.clone();

        for (int i = 0; i < clusterImage.rows; i++) {
            for (int j = 0; j < clusterImage.cols; j++)
            {
                if (clusterImage.at<int>(i, j) == 1 && (image_ref_roughness.at<float>(i,j) < roughness_threshold))
                {
                circle(inputImage_copy_1, Point(j, i), 1, Scalar(255, 0, 0), 1, 8, 0);
                circle(inputImage, Point(j, i), 1, Scalar(255, 0, 0), 1, 8, 0);
                }
                else if (clusterImage.at<int>(i,j)==2){
                circle(inputImage_copy_2, Point(j, i), 1, Scalar(0, 255, 0), 1, 8, 0);
                circle(inputImage, Point(j, i), 1, Scalar(0, 255, 0), 1, 8, 0);
                }else if (clusterImage.at<int>(i,j)==3){
                circle(inputImage_copy_3, Point(j, i), 1, Scalar(0, 0, 255), 1, 8, 0);
                circle(inputImage, Point(j, i), 1, Scalar(0, 0, 255), 1, 8, 0);
                }
            }
        }
        imshow("inputImage_specular_Distribution_1", inputImage_copy_1);
        imshow("inputImage_specular_Distribution_2", inputImage_copy_2);
        imshow("inputImage_specular_Distribution_3", inputImage_copy_3);
        clusterImage.convertTo(clusterImage, CV_8UC1, 255.0);
        //        imshow("Img_range",Img_range);
        //        imshow("diffuseImage", diffuseImage);
        //        imshow("specularImage", specularImage);
        //        imshow("Input Image", Image_ref8UC3);
        //        imshow("Output Image", diffuseImage);
        imshow("Specular_Diffuse_TransitionField", transitionField);
        imshow("DSO_selectedPointMask", selectedPointMask1);
        std::cerr<<"\n show point_counter:"<<point_counter<<endl;
        std::cerr<<"\n show dso_point_counter:"<<dso_point_counter<<endl;
        cout<<"\n Image averge:"<< mean_val<<endl;
        cout<<" Image std:"<<std_dev<<endl;
        waitKey(0);
        }




//    // ----------------------------------------optimization variable: R, t--------------------------------------
    Sophus::SE3d xi, xi_GT;
    Sophus::SE3d pose1_gt( dataLoader->R1, dataLoader->t1);
    Sophus::SE3d pose2_gt( dataLoader->R2, dataLoader->t2);
    Eigen::Matrix<double,3,3> R;
    R = dataLoader->q_12.normalized().toRotationMatrix();
    xi_GT.setRotationMatrix(R);
    xi_GT.translation() = dataLoader->t12;
    //-------------------------------------------------Data perturbation--------------------------------------------------------------------
    double trErr;
//    Eigen::Vector3d perturbedTranslation = translation_pertabation(0.3, 0.3, 0.3, T_GT, trErr); // percentage
    Eigen::Vector3d perturbedTranslation = translation_pertabation(0.0, 0.0, 0.0, xi_GT.translation(), trErr); // percentage
    Eigen::Vector3d t12_input = perturbedTranslation;

    double roErr;
//    Eigen::Matrix3d perturbedRotation = rotation_pertabation(8.0,8.0,8.0, R_GT, roErr); // degree
    Eigen::Matrix3d perturbedRotation = rotation_pertabation(0.0,0.0,0.0, xi_GT.rotationMatrix(), roErr); // degree
    Eigen::Quaterniond q12_input = Eigen::Quaterniond(perturbedRotation);
    q12_input.normalize();
    Sophus::SE3d inputPoseGT(xi_GT.rotationMatrix(), xi_GT.translation());

    cout << "\n Show initial rotation:\n" << q12_input.toRotationMatrix()<< "\n Show initial translation:\n" << t12_input<< endl;
    cout << "\n Show current rotation perturbation error :" << roErr<< "\n Show current translation perturbation error : " << trErr << "\nShow current depth perturbation error :"<< "depth_Error" << endl;


    // setup ceres problem
    size_t num_cameras = 2;
    Sophus::SE3d inputPose(q12_input,t12_input);

    bool useGT= false;
    bool useImagePyramid = true;
    bool useDelta = true;

    double camera_poses[]={1, 0, 0, 0, 0, 0, 0,1, 0, 0, 0, 0, 0, 0};
    if (useGT){
        double  camera_posesGT[14]= {1, 0, 0, 0, 0, 0, 0,q12_input.w(), q12_input.x(), q12_input.y(), q12_input.z(), t12_input[0], t12_input[1], t12_input[2]};
        std::memcpy(camera_poses, camera_posesGT, 14* sizeof(double));
    }
    // use image pyramid or not
    float huberPara=40.0f;
    int image_pyramid= 5;
    int lvl_target, lvl_ref;
    std::vector<cv::Point3f> points3D;
    Mat pointOfInterestArea(grayImage_ref.rows, grayImage_ref.cols, CV_8UC1, Scalar(0));

    if (useImagePyramid){

        Eigen::Matrix3f K=K_synthetic.cast<float>();
        for (int lvl = image_pyramid; lvl >= 1; lvl--) {
            points3D.clear();
            cout << "\n Show the value of lvl:" << lvl << endl;
            Mat IRef, DRef, I, D;
            Eigen::Matrix3f Klvl, Klvl_ignore;
            lvl_target = lvl;
            lvl_ref = lvl;
            int  npts_lvl_ref[lvl];
            int  npts_lvl_tar[lvl];
            float* statusMapPoints_ref= new float[wG[lvl-1]*hG[lvl-1]];
            float* statusMapPoints_tar= new float[wG[lvl-1]*hG[lvl-1]];
            //            downscale(grayImage_ref_green, depth_ref_inv, K, lvl_ref, IRef, DRef, Klvl);
            //            downscale(grayImage_tar_green, depth_tar_inv, K, lvl_target, I, D, Klvl_ignore);


            downscale(grayImage_ref, depth_ref_inv, K, lvl_ref, IRef, DRef, Klvl);
            downscale(grayImage_tar, depth_tar_inv, K, lvl_target, I, D, Klvl_ignore);
            // show size of IRef
            cout << "\n Show the size of IRef:" << IRef.size() << endl;

            if (usePixelSelector and lvl==1){
                FrameHessian* frame_ref= new FrameHessian();
                FrameHessian* frame_tar= new FrameHessian();
                PixelSelector* pixelSelector_lvl= new PixelSelector(wG[lvl-1],hG[lvl-1]);
                float* color_ref_lvl= new float[wG[lvl-1]*hG[lvl-1]];
                float* color_tar_lvl= new float[wG[lvl-1]*hG[lvl-1]];
                for (int row = 0; row < hG[lvl-1]; ++row) {
                    uchar *pixel_ref_lvl=IRef.ptr<uchar>(row);
                    uchar *pixel_tar_lvl=I.ptr<uchar>(row);
                    for (int col = 0; col < wG[lvl-1]; ++col) {
                        color_ref_lvl[row*wG[lvl-1]+col]= (float) pixel_ref_lvl[col];
                        color_tar_lvl[row*wG[lvl-1]+col]= (float) pixel_tar_lvl[col];
                    }
                }

                frame_ref->makeImages(color_ref_lvl);
                frame_tar->makeImages(color_tar_lvl);
                pixelSelector_lvl->currentPotential= 3;

                npts_lvl_ref[lvl-1]=  pixelSelector_lvl->makeMaps(frame_ref, statusMapPoints_ref, densities[lvl-1] * wG[0] * hG[0], 1, false, 2);
                cout << "\n npts_lvl_ref[i]: " << npts_lvl_ref[lvl-1] << "\n densities[i-1]*wG[0]*hG[0]:" << densities[lvl-1] * wG[0] * hG[0] << endl;
                npts_lvl_tar[lvl-1]=  pixelSelector_lvl->makeMaps(frame_tar, statusMapPoints_tar, densities[lvl-1] * wG[0] * hG[0], 1, false, 2);
                cout << "\n npts_lvl_tar[i]: " << npts_lvl_tar[lvl-1] << "\n densities[i-1]*wG[0]*hG[0]:" << densities[lvl-1] * wG[0] * hG[0] << endl;


                // check the dso selected points
                Mat dsoSelectedPointMask(grayImage_ref.rows,  grayImage_ref.cols, CV_8UC1, Scalar(0));
                Mat dsoSelectedPointAndISMask(grayImage_ref.rows,  grayImage_ref.cols, CV_8UC1, Scalar(0));


                int point_counter=0;
                int dso_point_counter=0;
                int dso_point_counter_merge=0;
                for (int u = 0; u< grayImage_ref.rows; u++) // colId, cols: 0 to 480
                {
                    for (int v = 0; v < grayImage_ref.cols; v++) // rowId,  rows: 0 to 640
                    {
                        if ((statusMapPoints_ref!=NULL && statusMapPoints_ref[u*grayImage_ref.cols+v]!=0) ){
                            dso_point_counter+=1;
                            dsoSelectedPointMask.at<uchar>(u,v)= 255;
                        }
                    }
                }

                imshow("puredsoSelectedPointMask", dsoSelectedPointMask);


                for (int u = 0; u< grayImage_ref.rows; u++) // colId, cols: 0 to 480
                {
                    for (int v = 0; v < grayImage_ref.cols; v++) // rowId,  rows: 0 to 640
                    {
                        if(image_ref_roughness.at<float>(u,v) < roughness_threshold && clusterImage.at<int>(u, v) == 1){
                            pointOfInterestArea.at<uchar>(u,v)= 255;
                        }


                        if (((statusMapPoints_ref!=NULL && statusMapPoints_ref[u*grayImage_ref.cols+v]!=0 )) || (image_ref_roughness.at<float>(u,v) < roughness_threshold && clusterImage.at<int>(u, v) == 1)){
                            dso_point_counter_merge+=1;
                            dsoSelectedPointAndISMask.at<uchar>(u,v)= 255;
                            statusMapPoints_ref[u*grayImage_ref.cols+v]=6;// 6 is not fixed
                        }
                    }
                }

                imshow("dsoSelectedPointMask", dsoSelectedPointAndISMask);
                cout << "\n dso_point_counter_merge: " << dso_point_counter_merge << endl;

//                pbaRelativePose(huberPara, IRef,statusMapPoints_ref,DRef, I,statusMapPoints_tar,Klvl.cast<double>(),camera_poses, points3D);
                // merge the transition field (less roughness)and dso selected points and use non-lambertian correction then optimize the pose , depth again
                // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!try to not use dso pixel selector but use the intensity segmentation method while using the dso selected points, information(lambertian+ corrected non-lambertian) merging

                if (useDelta){
                    imshow("pointOfInterestArea", pointOfInterestArea);
                    //	//--------------------------------------------------normal_map_GT---------------------------------------------------
                    cv::Mat normal_map(grayImage_ref.rows, grayImage_ref.cols, CV_32FC3);
                    for (int u = 0; u < grayImage_ref.rows; u++) // colId, cols: 0 to 480
                    {
                        for (int v = 0; v < grayImage_ref.cols; v++) // rowId,  rows: 0 to 640
                        {
                            Eigen::Vector3f normal_new(normal_map_GT.at<cv::Vec3f>(u, v)[0], normal_map_GT.at<cv::Vec3f>(u, v)[1],normal_map_GT.at<cv::Vec3f>(u, v)[2]);
                            //			normal_new = (dataLoader->R1.cast<float>()).transpose()* normal_new;
                            Eigen::Vector3f principal_axis(0, 0, 1);
                            if (normal_new.dot(principal_axis) > 0) {normal_new = -normal_new;}
                            normal_map.at<Vec3f>(u, v)[0] = normal_new(0);
                            normal_map.at<Vec3f>(u, v)[1] = normal_new(1);
                            normal_map.at<Vec3f>(u, v)[2] = normal_new(2);

                        }
                    }
                    imshow("normal_map",normal_map);
                    waitKey(0);
                    Mat newNormalMap = normal_map;
                    Mat deltaMap(grayImage_ref.rows, grayImage_ref.cols, CV_32FC3, Scalar(1,1,1)); // storing delta
                    Mat envMapWorkMask(deltaMap.rows, deltaMap.cols, CV_8UC1, Scalar(0));

                    Sophus::SO3d Rotation_GT(xi_GT.rotationMatrix());
                    Eigen::Matrix<double, 3, 1> Translation_GT(xi_GT.translation());
                    DSONL::updateDelta(dataLoader->camPose1,EnvLightLookup, statusMapPoints_ref,Rotation_GT,Translation_GT,K_synthetic,depth_ref_inv,image_ref_roughness,deltaMap,newNormalMap, pointOfInterestArea, renderedEnvMapPath,envMapWorkMask);

                    imshow("deltaMap",deltaMap);
                    imshow("envMapWorkMask",envMapWorkMask);
                    waitKey(0);

                }
//                for (int u = 0; u< grayImage_ref.rows; u++) // colId, cols: 0 to 480
//                {
//                    for (int v = 0; v < grayImage_ref.cols; v++) // rowId,  rows: 0 to 640
//                    {
//                        // || clusterImage.at<int>(u, v) == 1
//                        if (((statusMap!=NULL && statusMapPoints_ref[u*grayImage_ref.cols+v]!=0 )) || ((image_ref_roughness.at<float>(u,v) < roughness_threshold)&&clusterImage.at<int>(u, v) == 1)){
//                            dso_point_counter+=1;
//                            dsoSelectedPointAndISMask.at<uchar>(u,v)= 255;
//                            statusMapPoints_ref[u*grayImage_ref.cols+v]=6;// 6 is not fixed
//
//                            // apply the intensity segmentation here // && (grayImage_ref.at<uchar>(u,v)>(mean_val+scale_std*std_dev))
////                            if ( (image_ref_roughness.at<float>(u,v) < roughness_threshold)  && clusterImage.at<int>(u, v) == 1 ){
////                                transitionField.at<uchar>(u,v)= 255;
////                                statusMap[u*grayImage_ref.cols+v]=255;
////                                point_counter+=1;
////                            }
//                        }
//                    }
//                }
//
//                imshow("dsoSelectedPointAndISMask", dsoSelectedPointAndISMask);


                delete pixelSelector_lvl;
                delete frame_ref;
                delete frame_tar;
                delete[] statusMapPoints_ref;
                delete[] statusMapPoints_tar;
                delete[] color_ref_lvl;
                delete[] color_tar_lvl;

            }else{
//                pbaRelativePose(huberPara, IRef,DRef, I,Klvl.cast<double>(),camera_poses, points3D);
            }
        }



    } else{
        pbaRelativePose(huberPara, grayImage_ref_green,depth_ref_inv, grayImage_tar_green,K_synthetic.cast<double>(),camera_poses, points3D);
    }




    // result analysis
    Sophus::SE3d pose_2 = Sophus::SE3d(Eigen::Quaterniond(camera_poses[7], camera_poses[8], camera_poses[9], camera_poses[10]),
                                       Eigen::Vector3d(camera_poses[11], camera_poses[12], camera_poses[13]));

    std::cout << pose_2.matrix() << std::endl;
    cout << "\n Show optimized rotation:\n" << pose_2.rotationMatrix()<< std::endl;
    Eigen::Vector3d ea = pose_2.rotationMatrix().eulerAngles(0, 1, 2);
    cout << "to Euler angles(XYZ):" << endl;
    cout << ea*180/M_PI << endl << endl;
    cout<<"\n Show optimized translation:\n"<< pose_2.translation() << endl;


    cout << "\n Show Rotational error :" << rotationErr(xi_GT.rotationMatrix(),  pose_2.rotationMatrix())
         << "(degree)." << "\n Show translational error :"
         << 100 * translationErr(xi_GT.translation(), pose_2.translation()) << "(%) "
         << "\n Show depth error :" << "depthErr(depth_ref_gt, inv_depth_ref).val[0]"
         << endl;

    float fx_ = K_synthetic(0,0);
    float fy_ = K_synthetic(1,1);
    float cx_ = K_synthetic(0,2);
    float cy_ = K_synthetic(1,2);

    showPointCloud(inputPoseGT,inputPose, pose1_gt, pose2_gt, pose_2, points3D, fx_, fy_, cx_, cy_);








    // apply the intensity segmentation here // && (grayImage_ref.at<uchar>(u,v)>(mean_val+scale_std*std_dev))
//                            if ( (image_ref_roughness.at<float>(u,v) < roughness_threshold)  && clusterImage.at<int>(u, v) == 1 ){
//                                transitionField.at<uchar>(u,v)= 255;
//                                statusMap[u*grayImage_ref.cols+v]=255;
//                                point_counter+=1;
//                            }
















//    //    string pointOfInterest= "../data/Exp_specular_floor/point_of_Interest.txt";
//    //    string pointOfInterest= "../data/Exp_specular_floor/point_of_I2.txt";
//    //    readUV(pointOfInterest, pointOfInterestArea);
//
//    //    pointOfInterestArea= imread("../data/Exp_specular_floor_forLoss/leftImage/spointMask_38880.png", IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
//    //    pointOfInterestArea= imread("../data/Exp_specular_floor_forLoss/leftImage/sPointLambertianMask_30118.png", IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
//    //    pointOfInterestArea= imread("../data/Exp_specular_floor_forLoss/leftImage/sPointLambertianMask_17512.png", IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
//
//    //    Mat sPointLambertianMask_15402= imread("../data/Exp_specular_floor_forLoss/leftImage/sPointLambertianMask_15402.png", IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
//    Mat sPointLambertianMask_10011= imread("../data/Exp_specular_floor_forLoss/leftImage/sPointLambertianMask_10011.png", IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
//    Mat pointOfInterestArea_Non_Lambertian_2358= imread("../data/Exp_specular_floor_forLoss/leftImage/spointMask_non_Lambertian_2358.png", IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
//    Mat pointOfInterestArea_allPoints_38880= imread("../data/Exp_specular_floor_forLoss/leftImage/spointMask_38880.png", IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
//
//    int num_points_used =0;
//    for (int u = 0; u < grayImage_ref.rows; u++)
//    {
//        for (int v = 0; v < grayImage_ref.cols; v++)
//        {
////             if (static_cast<int>(sPointLambertianMask_10011.at<uchar>(u, v))==255 || static_cast<int>(pointOfInterestArea_Non_Lambertian_2358.at<uchar>(u, v))==255 ){
//            if (int(pointOfInterestArea_allPoints_38880.at<uchar>(u, v))==255 )
//            {
//                num_points_used+=1;
//                pointOfInterestArea.at<uchar>(u,v)= 255;
//            }
//
////            if (int(pointOfInterestArea_Non_Lambertian_2358.at<uchar>(u, v))==255 ) {
////                num_points_used+=1;
////                pointOfInterestArea.at<uchar>(u,v)= 255;
////            }
//
//
//        }
//    }
//    imshow("pointOfInterestArea",pointOfInterestArea);
//    cout<<"check channel of pointOfInterestArea_Non_Lambertian_2358:"<<pointOfInterestArea_Non_Lambertian_2358.channels()<<" check num_points_used "<<num_points_used<<endl;
////    waitKey(0);
//

//

//
//    // ====================================== pointSelector========================================
//    bool usePixelSelector= false;
//    float densities[] = {0.03,0.003, 0.05,0.15,0.5,1}; /// number of optimized depths,  current index is 1
//    PixelSelector* pixelSelector=NULL;
//    FrameHessian* newFrame_ref=NULL;
//    FrameHessian* newFrame_tar=NULL;
//    FrameHessian* depthMap_ref=NULL;
//    float* color_ref=NULL;
//    float* color_tar=NULL;
//    float* depthMapArray_ref=NULL;
//    float* statusMap=NULL;
//    bool*  statusMapB=NULL;
//    Mat statusMap_NonLambCand(grayImage_ref.rows,  grayImage_ref.cols, CV_8UC1, Scalar(0));
//
//    if (usePixelSelector){
//        double min_gray,max_gray;
//        Mat grayImage_ref_CV8U;
//        Mat grayImage_tar_CV8U;
//        imshow("grayImage_selector_ref",dataLoader->grayImage_selector_ref);
//        dataLoader->grayImage_selector_ref.convertTo(grayImage_ref_CV8U,CV_8UC1, 255.0);
////        grayImage_target.convertTo(grayImage_tar_CV8U,CV_8UC1, 255.0);
//
//        imshow("grayImage_ref_CV8U",grayImage_ref_CV8U);
//        waitKey(0);
//        newFrame_ref= new FrameHessian();
//        newFrame_tar= new FrameHessian();
//
//
//        pixelSelector= new PixelSelector(wG[0],hG[0]);
//        color_ref= new float[wG[0]*hG[0]];
//        color_tar= new float[wG[0]*hG[0]];
//
//
//        for (int row = 0; row < hG[0]; ++row) {
//
//            uchar *pixel_ref=grayImage_ref_CV8U.ptr<uchar>(row);
//            uchar *pixel_tar=grayImage_ref_CV8U.ptr<uchar>(row);
////            float * pixel_depth_ref= inv_depth_ref.ptr<float>(row);
//
//            for (int col = 0; col < wG[0]; ++col) {
//                color_ref[row*wG[0]+col]= (float) pixel_ref[col];
//                color_tar[row*wG[0]+col]= (float)pixel_tar[col];
////              depthMapArray_ref[row*wG[0]+col]=pixel_depth_ref[col];
//
//            }
//        }
//        newFrame_ref->makeImages(color_ref); // make image_ref pyramid
//        newFrame_tar->makeImages(color_tar); // make image_tar pyramid
//        statusMap= new float[wG[0]*hG[0]];
//        statusMapB = new bool[wG[0]*hG[0]];
//        int setting_desiredImmatureDensity=1500;
//        float densities[] = {1,0.5,0.15,0.05,0.03}; // 不同层取得点密度
//
//        int  npts[pyrLevelsUsed];
//        Mat selectedPointMask1(grayImage_ref.rows,  grayImage_ref.cols, CV_8UC1, Scalar(0));
//
//
//        // MinimalImageB3 imgShow[pyrLevelsUsed];
//        pixelSelector->currentPotential= 3;
//        for(int i=0; i>=0; i--) {
//
//            cout << "\n pyrLevelsUsed:" << i << endl;
//			plotImPyr(newFrame_ref, i, "newFrame_ref");
////			plotImPyr(newFrame_tar, i, "newFrame_tar");
////			plotImPyr(depthMap_ref, i, "depthMap_ref");
//            npts[i] = pixelSelector->makeMaps(newFrame_ref, statusMap, densities[1] * wG[0] * hG[0], 1, true, 2);
//            cout << "\n npts[i]: " << npts[i] << "\n densities[i]*wG[0]*hG[0]:" << densities[i] * wG[0] * hG[0] << endl;
//            waitKey(0);
//
////            cv::Mat image_ref(hG[i], wG[i], CV_32FC1);
////            memcpy(image_ref.data, newFrame_ref->img_pyr[i], wG[i] * hG[i] * sizeof(float));
////            cv::Mat image_tar(hG[i], wG[i], CV_32FC1);
////            memcpy(image_tar.data, newFrame_tar->img_pyr[i], wG[i] * hG[i] * sizeof(float));
//        }
//
//
//        float metallic_threshold = 0.8;
//        float roughness_threshold = 0.3;
////        float scale_std= 0.6; // LDR
//        float scale_std= 0.6; // HDR maybe wrong
////        float scale_std= 0.8; // HDR only for test
//
//
//
////        show point_counter:720
////        show dso_point_counter:38878
////        Image averge:0.360396
////        Image std:0.521962
//
//
//        int point_counter=0;
//        int dso_point_counter=0;
//
//        for (int u = 0; u< grayImage_ref.rows; u++) // colId, cols: 0 to 480
//        {
//            for (int v = 0; v < grayImage_ref.cols; v++) // rowId,  rows: 0 to 640
//            {
//                if (statusMap!=NULL && statusMap[u*grayImage_ref.cols+v]!=0 ){
//                    dso_point_counter+=1;
//
//                    // ================================save the selectedPoint mask here=======================
//
//                    if ( (image_ref_roughness.at<float>(u,v) < roughness_threshold || image_ref_metallic.at<float>(u,v)>metallic_threshold) && (grayImage_ref.at<double>(u,v)>(dataLoader->mean_val+scale_std*dataLoader->std_dev) )){
//                        statusMap_NonLambCand.at<uchar>(u,v)= 255;
//                        statusMap[u*grayImage_ref.cols+v]=255;
//                        point_counter+=1;
//                    }
//
//
////                    if ( (image_ref_roughness.at<float>(u,v) < roughness_threshold || image_ref_metallic.at<float>(u,v)>metallic_threshold) && (dataLoader->grayImage_ref_CV8UC1.at<uchar>(u,v)>(dataLoader->mean_val+scale_std*dataLoader->std_dev) )){
////                        statusMap_NonLambCand.at<uchar>(u,v)= 255;
////                        statusMap[u*grayImage_ref.cols+v]=255;
////                        point_counter+=1;
////                    }
//
//                }
//            }
//        }
//        // refine the point selector
////        imshow("selectedPointMask1",selectedPointMask1);
//        imshow("statusMap_NonLambCand", statusMap_NonLambCand);
//
//        std::cerr<<"\n show point_counter:"<<point_counter<<endl;
//        std::cerr<<"\n show dso_point_counter:"<<dso_point_counter<<endl;
//        cout<<"\n Image averge:"<< dataLoader->mean_val<<endl;
//        cout<<" Image std:"<<dataLoader->std_dev<<endl;
////        imwrite("pointMask.png", statusMap_NonLambCand);
////        imwrite("selectedPointMask1.png",selectedPointMask1);
//        waitKey(0);
//
//        }
//
//

//    imshow("grayImage_target", grayImage_target);
//    waitKey(0);
//	// show the depth image with noise
//	double min_depth_val, max_depth_val;
//	cv::minMaxLoc(depth_ref, &min_depth_val, &max_depth_val);
//	cout << "\n show original depth_ref min, max:\n" << min_depth_val << "," << max_depth_val << endl;
//
//	// grayImage_ref
//    double min_radiance_val, max_radiance_val;
//    cv::minMaxLoc(grayImage_ref, &min_radiance_val, &max_radiance_val);
//    cout << "\n show original grayImage_ref min, max:\n" << min_radiance_val << "," << max_radiance_val << endl;
//
//
////	imshow("grayImage_ref",grayImage_ref);
////	imshow("grayImage_target",grayImage_target);
////	waitKey(0);
//
//	// ----------------------------------------optimization variable: depth --------------------------------------
//	cout << "\n Show GT rotation:\n" << xi_GT.rotationMatrix() << "\n Show GT translation:\n" << xi_GT.translation()<< endl;
//	// -------------------------------------------------Movingleast algorithm-----------------------------------------
//	std::vector<Eigen::Vector3d> pts;

//	//MLS();

//
//    //	//-------------------------------------------------Data perturbation--------------------------------------------------------------------
//    //	// Add noise to original depth image, depth_ref_NS
//	Mat inv_depth_ref, depth_ref_gt;
//	Mat depth_ref_NS;
//	double roErr;
//	Eigen::Matrix3d R_GT(xi_GT.rotationMatrix());
//	Eigen::Matrix3d perturbedRotation = rotation_pertabation(0.0, 0.0, 0.0, R_GT, roErr); // degree
//
//	double trErr;
//	Eigen::Vector3d T_GT(xi_GT.translation());
//	Eigen::Vector3d perturbedTranslation = translation_pertabation(0.0, 0.0, 0.0, T_GT, trErr); // percentage
//	double Mean = 0.0, StdDev = 0;
//	//	float densities[] = {0.03, 0.003, 0.05, 0.15, 0.5, 1}; /// number of optimized depths,  current index is 1
//

//
//	PhotometricBAOptions options;

////    Mat newNormalMap = normal_map_GT;
////    double distanceThres = 0.009;
////	double distanceThres = 0.007;
//    double distanceThres = 0.0035;
////    double distanceThres = 0.002;
//
//	float upper = 2.0;
//	float buttom = 0.5;
//	float up_new = upper;
//	float butt_new = buttom;


//
//
//    Mat deltaMap_GT(depth_ref.rows, depth_ref.cols, CV_32FC1, Scalar(1)); // storing delta
//    Mat deltaRatio(depth_ref.rows, depth_ref.cols, CV_32FC1, Scalar(1)); // storing delta
//
//	int lvl_target, lvl_ref;
//
//	double depth_upper_bound = 0.5;  // 0.5; 1
//	double depth_lower_bound = 0.1;  // 0.001
//
//	options.optimize_depth = false;
//	options.useFilterController = false; // control the number of optimized depth
//	options.optimize_pose = true;
//	options.use_huber = false;
//	options.lambertianCase = false;
//	options.usePixelSelector = false;
//	dataLoader->options_.remove_outlier_manually = false;
//	options.huber_parameter = 0.25*4.0 / 255.0;   /// 0.25*4/255 :   or 4/255
//    bool useImgPyramid = true;
//	// -----------------------------------------Initialize the pose xi with GT or just use the default value---------------------
////	Eigen::Vector3d initial_translation;
////	initial_translation<< -0.2, -0.1, 0;
//
//	xi.setRotationMatrix(perturbedRotation);
//	xi.translation() = perturbedTranslation;
//
////	xi.translation() = initial_translation;
//
//	Sophus::SO3d Rotation(xi.rotationMatrix());
//	Eigen::Matrix<double, 3, 1> Translation(xi.translation());
//

//
//
//
//	AddGaussianNoise_Opencv(depth_ref, depth_ref_NS, Mean, StdDev, statusMap);
//	divide(Scalar(1), depth_ref, depth_ref_gt);
//	divide(Scalar(1), depth_ref_NS, inv_depth_ref);
//	Mat inv_depth_tar;
//	divide(Scalar(1), depth_target, inv_depth_tar);
//
//
//	Mat depth_ref_NS_before = inv_depth_ref.clone();
//	double min_inv, max_inv;
//	cv::minMaxLoc(inv_depth_ref, &min_inv, &max_inv);
//	cout << "\n show original inv_depth_ref min, max:\n" << min_inv << "," << max_inv << endl;
//	Scalar_<double> depth_Err = depthErr(depth_ref_gt, inv_depth_ref);
//	double depth_Error = depth_Err.val[0];
//	cout << "\n Show initial rotation:\n" << Rotation.matrix() << "\n Show initial translation:\n" << Translation<< endl;
//	cout << "\nShow current rotation perturbation error :" << roErr<< "\n Show current translation perturbation error : " << trErr << "\nShow current depth perturbation error :"<< depth_Error << endl;
//
//	double min_gt_special, max_gt_special;
//	cv::minMaxLoc(inv_depth_ref, &min_gt_special, &max_gt_special);
//	cout << "\n show inv_depth_ref min, max:\n" << min_gt_special << "," << max_gt_special << endl;
//	//	Mat inv_depth_ref_for_show = inv_depth_ref * (1.0 / (max_gt_special - min_gt_special)) +(-min_gt_special * (1.0 / (max_gt_special - min_gt_special)));
//	//	string depth_ref_name = "inv_depth_ref";
//	//	imshow(depth_ref_name, inv_depth_ref_for_show);
//
//
//
//
//	for (int lvl = 1; lvl >= 1; lvl--) {
//		cout << "\n Show the value of lvl:" << lvl << endl;
//		Mat IRef, DRef, I, D;
//		Eigen::Matrix3f Klvl, Klvl_ignore;
//		lvl_target = lvl;
//		lvl_ref = lvl;
//
//		downscale(grayImage_ref, inv_depth_ref, K, lvl_ref, IRef, DRef, Klvl);
//		downscale(grayImage_target, depth_target, K, lvl_target, I, D, Klvl_ignore);
//		double min_gt_special, max_gt_special;
//		int i = 0;
//		while (i < 2) {
//			double max_n_, min_n_;
//			cv::minMaxLoc(deltaMap, &min_n_, &max_n_);
//			cout << "->>>>>>>>>>>>>>>>>show max and min of estimated deltaMap:" << max_n_ << "," << min_n_ << endl;
//            Mat mask = cv::Mat(deltaMap != deltaMap);
//			deltaMap.setTo(1.0, mask);
//			if (i == 1) {
//				cout << "depthErr(depth_ref_gt, inv_depth_ref).val[0]:" << depthErr(depth_ref_gt, inv_depth_ref).val[0]<< endl;
//				showScaledImage(depth_ref_NS_before, depth_ref_gt, inv_depth_ref);
//			}
//			cv::minMaxLoc(inv_depth_ref, &min_gt_special, &max_gt_special);
//			cout << "\n show inv_depth_ref min, max:\n" << min_gt_special << "," << max_gt_special << endl;
//			Mat inv_depth_ref_for_show = inv_depth_ref * (1.0 / (max_gt_special - min_gt_special)) +(-min_gt_special * (1.0 / (max_gt_special - min_gt_special)));
//			string depth_ref_name = "inv_depth_ref" + to_string(i);
////			imshow(depth_ref_name, inv_depth_ref_for_show);
////			cout<<"show the current depth:"<<inv_depth_ref.at<double>(359,470)<<endl;
//            // Photometric loss
//            statusMap_NonLambCand = pointOfInterestArea.clone();
////            PhotometricBA(IRef, I, options, Klvl, Rotation, Translation, inv_depth_ref, deltaMap, depth_upper_bound,depth_lower_bound, statusMap, statusMapB,statusMap_NonLambCand);
//
//            // Result analysis
//            cout << "\n show depth_ref min, max:\n" << min_gt_special << "," << max_gt_special << endl;
//            cout << "\n Show optimized rotation:\n" << Rotation.matrix() << "\n Show optimized translation:\n"<< Translation << endl;
//            cout << "\n Show Rotational error :" << rotationErr(xi_GT.rotationMatrix(), Rotation.matrix())
//                 << "(degree)." << "\n Show translational error :"
//                 << 100 * translationErr(xi_GT.translation(), Translation) << "(%) "
//                 << "\n Show depth error :" << depthErr(depth_ref_gt, inv_depth_ref).val[0]
//                 << endl;
//            std::cout << "\n Start calculating delta map ... " << endl;
//            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
//
//
//            // use estimated pose
////            DSONL::updateDelta(dataLoader->camPose1,EnvLightLookup, statusMap,Rotation,Translation,Klvl,image_ref_baseColor,inv_depth_ref,image_ref_metallic ,image_ref_roughness,deltaMap,newNormalMap,up_new, butt_new, pointOfInterestArea_Non_Lambertian_2358);//         DSONL::updateDelta(dataLoader->camPose1,EnvLightLookup, statusMap,Rotation,Translation,Klvl,image_ref_baseColor,inv_depth_ref,image_ref_metallic ,image_ref_roughness,deltaMap,newNormalMap,up_new, butt_new);
//            // use GT  pose
////            DSONL::updateDelta(dataLoader->camPose1,EnvLightLookup, statusMap,Rotation_GT,Translation_GT,Klvl,image_ref_baseColor,inv_depth_ref,image_ref_metallic ,image_ref_roughness,deltaMap,newNormalMap,up_new, butt_new, pointOfInterestArea_allPoints_38880, renderedEnvMapPath);
////            DSONL::updateDelta(dataLoader->camPose1,EnvLightLookup, statusMap,Rotation_GT,Translation_GT,Klvl,image_ref_baseColor,inv_depth_ref,image_ref_metallic ,image_ref_roughness,deltaMap,newNormalMap,up_new, butt_new, pointOfInterestArea_Non_Lambertian_2358, renderedEnvMapPath);
//
//            DSONL::updateDelta(dataLoader->camPose1,EnvLightLookup, statusMap,Rotation_GT,Translation_GT,Klvl,image_ref_baseColor,inv_depth_ref,
//                               image_ref_metallic ,image_ref_roughness,deltaMap,newNormalMap,up_new, butt_new, pointOfInterestArea, renderedEnvMapPath
//                               ,envMapWorkMask);
//
//
//
//            // deltaMapGT
////            Mat deltaMapGT_res= deltaMapGT(grayImage_ref,depth_ref,grayImage_target,depth_target,K.cast<double>(),distanceThres,xi_GT, upper, buttom, deltaMap, statusMap, pointOfInterestArea_allPoints_38880);
////            Mat deltaMapGT_res= deltaMapGT(grayImage_ref,depth_ref,grayImage_target,depth_target,K.cast<double>(),distanceThres,xi_GT, upper, buttom, deltaMap, statusMap, pointOfInterestArea_Non_Lambertian_2358);
//            Mat deltaMapGT_res= deltaMapGT(grayImage_ref,depth_ref,grayImage_target,depth_target,K.cast<double>(),distanceThres,xi_GT,
//                                           upper, buttom, deltaMap, statusMap, envMapWorkMask,
//                                           controlPointPose_path,
//                                           dataLoader->camPose1.cast<float>(),
//                                           newNormalMap
//                                           );
//
//
////            for (int u = 0; u < grayImage_ref.rows; u++)// colId, cols: 0 to 480
////            {
////                for (int v = 0; v < grayImage_ref.cols; v++)// rowId,  rows: 0 to 640
////                {
////                        if ( static_cast<int>(pointOfInterestArea_Non_Lambertian_2358.at<uchar>(u, v))==255 ){
//////                    if (static_cast<int>(pointOfInterestArea_Lambertian_14976.at<uchar>(u, v))==255 ){
////                        num_points_used+=1;
////                        pointOfInterestArea.at<uchar>(u,v)= 255;
////                    }
////                }
////            }
//
////            imshow("deltaMapGT_res",deltaMapGT_res);
////            waitKey(0);
//
//
//            // compare deltaMap_GT with deltaMap:
////            int counter_true=0, counter_all_pts=0;
////            for (int u = 0; u < grayImage_ref.rows; u++)// colId, cols: 0 to 480
////            {
////                for (int v = 0; v < grayImage_ref.cols; v++)// rowId,  rows: 0 to 640
////                {
////                    if (pointOfInterestArea_Non_Lambertian_2358.at<uchar>(u,v)!=255){ continue;}
////                    counter_all_pts+=1;
//////                    cout<<"check deltaMap_GT vals:"<<deltaMapGT_res.at<float>(u,v)<<"and, deltaMap.at<float>(u,v):"<<deltaMap.at<float>(u,v)<<endl;
////                    if ((deltaMapGT_res.at<float>(u,v)>=1.0 && deltaMap.at<float>(u,v)>=1.0) || (deltaMapGT_res.at<float>(u,v)<1.0 && deltaMap.at<float>(u,v)<1.0) ){
////                        counter_true+=1;
////                    }
////                    }
////                }
////        Mat mat_mean, mat_stddev;
////        double mean_val, std_dev;
////        meanStdDev(deltaRatio, mat_mean, mat_stddev);
////        mean_val= mat_mean.at<double>(0,0);
////        std_dev = mat_stddev.at<double>(0,0);
////        std::cerr<<"show counter_all_pts and delta_true_ratio "<< counter_all_pts<<"and num of counter_true:"<<counter_true<<"show ratio:"<<   (float)counter_true/ (float)counter_all_pts<<endl;
//
//
//            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
//            std::chrono::duration<double> time_used =std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
//            std::cout << "\n Delta map is done ... "<< " and costs time:" << time_used.count() << " seconds." << endl;
////			Mat deltaMapGT_res= deltaMapGT(grayImage_ref,depth_ref,grayImage_target,depth_target,K.cast<double>(),distanceThres,xi_GT, upper, buttom, deltaMap);
////
////          Mat showGTdeltaMap=colorMap(deltaMapGT_res, upper, buttom);
////          Mat showESdeltaMap=colorMap(deltaMap, upper, buttom);
////
////			imshow("show GT deltaMap", showGTdeltaMap);
////			imshow("show ES deltaMap", showESdeltaMap);
////			/// TEMP TEST BEGIN
////			for (int x = 0; x < deltaMapGT_res.rows; ++x) {
////				for (int y = 0; y < deltaMapGT_res.cols; ++y) {
////					if (deltaMapGT_res.at<float>(x, y) == -1) {deltaMap.at<float>(x, y) =1;}
////				}
////			}
////			double max_n_1, min_n_1;
////			cv::minMaxLoc(deltaMapGT_res, &min_n_1, &max_n_1);
////			cout << "->>>>>>>>>>>>>>>>>show max and min of deltaMapGT_res<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<:" << max_n_1 << "," << min_n_1 << endl;
//			//->>>>>>>>>>>>>>>>>show max and min of deltaMapGT_res<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<:8.7931,0.141176
////			 deltaMap=deltaMapGT_res.clone();// !!!!!!!!!!!!!!!!!!!!!!!!!!!!test!!!!!!!!!!!!!!!!!!!!!!!!!!!11
////            /// TEMP TEST END
////			imwrite("GT_deltaMap.exr", showGTdeltaMap);
////			imwrite("ES_deltaMap.exr", showESdeltaMap);
//			i += 1;
//
//		}
//		cout << "\nShow current rotation perturbation error :" << roErr
//		     << "\nShow current translation perturbation error : " << trErr
//		     << "\nShow current depth perturbation error :" << depth_Error << endl;
//
//
//
//		waitKey(0);
//	}
//
//	// tidy up
	delete dataLoader;
	if (usePixelSelector) {
		delete pixelSelector;
		delete newFrame_ref;
		delete newFrame_tar;
		delete depthMap_ref;
		delete[] statusMap;
		delete[] color_ref;
		delete[] color_tar;
		delete[] depthMapArray_ref;
		delete[] statusMap;
		delete[] statusMapB;
	}
	return 0;
}





// notes:

//
//std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
//std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
//std::chrono::duration<double> time_used =std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
//cout << "construct the envMap: " << time_used.count() << " seconds." << endl;


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


//    cout<<"\n show new outside envMapPose_world:\n "<< EnvLight->envLightMap[key4Search].envMapPose_world.matrix()<<endl;
//     cout<<"show EnvLight->brdfSampler size:"<< EnvLight->brdfSampler.size()<<endl;
//     cout<<"show size of EnvLight->envLightMap[firstPoint].EnvmapSampler size:"<<EnvLight->envLightMap[key4Search].EnvmapSampler.size()<<endl;
//    gli::vec4 SampleAAAAAA =EnvLight->envLightMap[key4Search].EnvmapSampler[0].texture_lod(gli::fsampler2D::normalized_type(0.6f,0.8f), 0.0f); // transform the texture coordinate
//    cout << "\n============SampleAAAAAA val(RGBA):\n"<< SampleAAAAAA.b << "," << SampleAAAAAA.g << "," <<SampleAAAAAA.r << "," << SampleAAAAAA.a << endl;

//    // Load Control pointCloud into Kd-tree
//    string control_pointCloud_path= "../include/EnvLight_Data/controlPoints/origins.pcd";
//
//    pcl::PointCloud<pcl::PointXYZ>::Ptr control_pointCloud (new pcl::PointCloud<pcl::PointXYZ>);
//
////    if (pcl::io::loadPCDFile<pcl::PointXYZ> (control_pointCloud_path, *control_pointCloud) == -1)
////      {
////           PCL_ERROR ("Couldn't read file control_pointCloud.pcd \n");
////           return (-1);
////      }
////    std::cout << "Number of loaded data points:"<< control_pointCloud->width * control_pointCloud->height<< std::endl;
////
////
////    for (const auto& point: *control_pointCloud)
////            std::cout << "    " << point.x
////                      << " "    << point.y
////                      << " "    << point.z << std::endl;
////
//
//    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
//    kdtree.setInputCloud(control_pointCloud);
//
//
//    // Load EnvMap into Unordered-map
//    std::unordered_map<cv::Point3f, int, hash3d<cv::Point3f>, equalTo<cv::Point3f>> envLightMap;









//void GetFileNames(string path,vector<string>& filenames , vector<string>& filenamesNoPath)
//{
//    DIR *pDir;
//    struct dirent* ptr;
//    if(!(pDir = opendir(path.c_str()))){
//        cout<<"Folder doesn't Exist!"<<endl;
//        return;
//    }
//    while((ptr = readdir(pDir))!=0) {
//        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
//            filenames.push_back(path + "/" + ptr->d_name);
//            filenamesNoPath.push_back(ptr->d_name);
//        }
//    }
//    closedir(pDir);
//}





// load env light maps
//    std::string envMap_Folder="../data/SimulationEnvData/envMap_10To16";
//    std::string envMap_Folder="/home/lei/Documents/Research/envMapData/envMap_10To16";

//    std::string envMap_Folder="/home/lei/Documents/Research/envMapData/envMapData_Dense01";
//    string controlPointPose_path= "/home/lei/Documents/Research/envMapData/envMapData_Dense0704_01_control_cam_pose3k.txt";

//    std::string envMap_Folder="/home/lei/Documents/Research/envMapData/ThirtyPointsEnvMap";
//    string controlPointPose_path= "/home/lei/Documents/Research/envMapData/short0370_02_control_cam_pose.txt";


//    std::string envMap_Folder="/home/lei/Documents/Research/envMapData/SeventeenPointsEnvMap";
//    string controlPointPose_path= "/home/lei/Documents/Research/envMapData/scene0370_02_control_cam_pose.txt";

//    std::string envMap_Folder=    "/home/lei/Documents/Research/envMapData/EnvMap_2358";
//    string controlPointPose_path= "/home/lei/Documents/Research/envMapData/2frame0370_02_control_cam_pose_2358.txt";
//    string  renderedEnvMapPath=  "/home/lei/Documents/Research/envMapData/EnvMap_2358";

//    std::string envMap_Folder="/home/lei/Documents/Research/envMapData/EnvMap_764";
//    string controlPointPose_path= "/home/lei/Documents/Research/envMapData/control_cam_pose_ControlpointCloud_Sparsfied_764.txt";
//    string  renderedEnvMapPath=   "/home/lei/Documents/Research/envMapData/EnvMap_764";

//    std::string envMap_Folder="/home/lei/Documents/Research/envMapData/EnvMap_91";
//    string controlPointPose_path= "/home/lei/Documents/Research/envMapData/control_cam_pose_ControlpointCloud_Sparsfied_91.txt";
//    string  renderedEnvMapPath=   "/home/lei/Documents/Research/envMapData/EnvMap_91";

//imshow("grayImage_ref",grayImage_ref);
//imshow("grayImage_target",grayImage_target);
//imshow("depth_ref",depth_ref_inv);
//imshow("depth_ref_GT",depth_ref_GT);
//waitKey(0);
//



//                Mat depth_ref = imread(depth_ref_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
//                Mat depth_reference(depth_ref.rows, depth_ref.cols, CV_64FC1);
//                for (int j = 0; j < depth_ref.rows; ++j) {
//                    for (int i = 0; i < depth_ref.cols; ++i) {
//                        depth_reference.at<double>(j,i)= 1.0/5000.0 * ((double ) depth_ref.at<unsigned short >(j,i));
////                        cout << "\n show  depth_reference: " << depth_reference.at<double>(j,i)<<endl;
//                    }
//                }



//                Mat depth_target = imread(depth_target_path, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
//                Mat depth_tar(depth_target.rows, depth_target.cols, CV_64FC1);
//
//                for (int j = 0; j < depth_target.rows; ++j) {
//                    for (int i = 0; i < depth_target.cols; ++i) {
//                        depth_tar.at<double>(j,i)= 1.0/5000.0 * ((double) depth_target.at<unsigned short >(j,i));
////                        cout << "\n show  depth_target: " << depth_tar.at<float>(j,i)<<endl;
//                    }
//                }


// check intensity segmentation
//// iterate through the Img_mask to find the location of the specular highlight
//Mat inputImage= Image_ref8UC3.clone();
//Mat inputImage_copy_1= inputImage.clone();
//Mat inputImage_copy_2= inputImage.clone();
//Mat inputImage_copy_3= inputImage.clone();
//
//
//for (int i = 0; i < clusterImage.rows; i++) {
//for (int j = 0; j < clusterImage.cols; j++) {
//if (clusterImage.at<int>(i, j) == 1) {
//circle(inputImage_copy_1, Point(j, i), 1, Scalar(255, 0, 0), 1, 8, 0);
//circle(inputImage, Point(j, i), 1, Scalar(255, 0, 0), 1, 8, 0);
//}
//else if (clusterImage.at<int>(i,j)==2){
//circle(inputImage_copy_2, Point(j, i), 1, Scalar(0, 255, 0), 1, 8, 0);
//circle(inputImage, Point(j, i), 1, Scalar(0, 255, 0), 1, 8, 0);
//}else if (clusterImage.at<int>(i,j)==3){
//circle(inputImage_copy_3, Point(j, i), 1, Scalar(0, 0, 255), 1, 8, 0);
//circle(inputImage, Point(j, i), 1, Scalar(0, 0, 255), 1, 8, 0);
//}
//}
//}
//
//imshow("inputImage_specular_Distribution_1", inputImage_copy_1);
//imshow("inputImage_specular_Distribution_2", inputImage_copy_2);
//imshow("inputImage_specular_Distribution_3", inputImage_copy_3);
//
//clusterImage.convertTo(clusterImage, CV_8UC1, 255.0);
//imshow("Img_range",Img_range);
//imshow("diffuseImage", diffuseImage);
//imshow("specularImage", specularImage);
//cv::imshow("Input Image", Image_ref8UC3);
//cv::imshow("Output Image", diffuseImage);
//waitKey(0);
//
//



