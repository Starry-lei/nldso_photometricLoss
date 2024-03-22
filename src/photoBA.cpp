//
// Created by lei on 28.04.23.
//

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "PhotometricBA/pba.h"
#include "PhotometricBA/utils.h"
#include "PhotometricBA/debug.h"
#include "PhotometricBA/dataset.h"
#include "PhotometricBA/imgproc.h"
#include "PhotometricBA/pose_utils.h"
#include <Eigen/Geometry>
#include <Eigen/Core>
#include  <signal.h>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <vector>
#include "gui_helper.h"
#include <chrono>
#include <iostream>
#include <thread>

struct CamWithId
{
	Mat44 pose; // 4x4 rigid body pose
	int  id;    // associated frame id
}; // PoseWithId

EigenAlignedContainer_<CamWithId> Init_traj_data;
EigenAlignedContainer_<CamWithId> Init_traj_data_from_relativePose;
void draw_scene( PhotometricBundleAdjustment::Result & res, EigenAlignedContainer_<CamWithId> & Init_traj_data, EigenAlignedContainer_<CamWithId> & Init_traj_data_from_relativePose);
pangolin::Var<bool> show_trajectory("ui.show_trajectory", true, true);
bool next_step(std::vector<std::shared_ptr<pangolin::ImageView>>& img_view);

string type2str(int type) ;
constexpr int UI_WIDTH = 200;
constexpr int NUM_CAMS = 1;
void draw_image_overlay(pangolin::View& v, size_t view_id);
pangolin::Var<bool> ui_show_hidden("ui.show_extra_options", false, true);
pangolin::Var<bool> show_cameras3d("hidden.show_cameras", true, true);
pangolin::Var<bool> continue_next("ui.continue_next", false, true);
pangolin::Var<bool> show_points3d("hidden.show_points", true, true);
pangolin::Var<bool> show_old_points3d("hidden.show_old_points3d", true, true);


bool gStop = false;
void sigHandler(int) { gStop = true; }
UniquePointer<Dataset> dataset;
PoseList T_init;
PoseList T_init_abs_pose ;
PhotometricBundleAdjustment::Result result;
PhotometricBundleAdjustment* photoba=nullptr;
EigenAlignedContainer_<Vec3> allRefinedPoints;
int fid=0; // start frame id

void readCtrlPointPoseData(string fileName, vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>>& pose) {

	ifstream trajectory(fileName);
	if (!trajectory.is_open()) {
		cout << "No controlPointPose data!" << fileName << endl;
		return;
	}

	float  qw, qx, qy, qz, tx, ty, tz;
	string line;
	while (getline(trajectory, line)) {
		stringstream lineStream(line);
		lineStream >> qw >> qx >> qy >> qz>> tx >> ty >> tz;

		Eigen::Vector3f t(tx, ty, tz);
		Eigen::Quaternionf q = Eigen::Quaternionf(qw, qx, qy, qz).normalized();
		Sophus::SE3f SE3_qt(q, t);
		pose.push_back(SE3_qt);



	}



}


int main(int argc, char** argv)
{
	bool show_gui = true;
    signal(SIGINT, sigHandler);
	pbaUtils::ProgramOptions options;

//  utils::ProgramOptions options;
    options
            ("output,o", "refined_poses_es_absolute.txt", "trajectory output file")
            ("config,c", "../config/tum_rgbd.cfg", "config file")
            .parse(argc, argv);

	pbaUtils::ConfigFile cf(options.get<std::string>("config"));

	dataset = Dataset::Create(options.get<std::string>("config"));



	// check if dataset is empty
	if(dataset == nullptr) {
		std::cerr<<"Failed to create dataset\n";
		return -1;
	}

	// load initial trajectory
	T_init = loadPosesTumRGBDFormat(cf.get<std::string>("trajectory"));
	//	T_init = loadPosesKittiFormat(cf.get<std::string>("trajectory"));
	// load GT trajectory
	//	std::string abs_pose= "../data/dataSetPBA_init_poor/Kitti_GT_00.txt";
	//	std::string abs_pose= "../data/dataSetPBA_init_poor/GT_pose_list_fr3.txt";
//	std::string abs_pose= "../data/dataSetPBA_init_poor/seq15/GT_Trajectory_seq15_650frames_WorldAtFirstFrame.txt";
//	std::string abs_pose= "../data/dataSetPBA_init_poor/seq16/GT_Trajectory_seq116_650_WorldFirst.txt";
	std::string abs_pose= "../data/dataSetPBA_init_poor/seq_17/GT_Trajectory_seq17_650_WorldFirst.txt";

//	std::string abs_pose= "../data/dataSetPBA_init_poor/seq_17/GT_Trajectory_seq15_650_WorldFirst.txt";
//	std::string abs_pose= "../data/dataSetPBA_init_poor/seq12_111_Poses_gt.txt";
//	std::string abs_pose= "../data/dataSetPBA_init_poor/scene0370_02_seq_01_tumRGBD_segmented_reseted.txt";
	cout<<"dataset created! "<<endl;

	T_init_abs_pose = loadPosesTumRGBDFormat(abs_pose);

	//	T_init_abs_pose = loadPosesKittiFormat(abs_pose);
	std::cout<<"trajectory: "<<cf.get<std::string>("trajectory")<<std::endl;
    if(T_init.empty()) {
        std::cerr<<("Failed to load poses from %s\n", cf.get<std::string>("trajectory").c_str());
        return -1;
    }

	photoba = new PhotometricBundleAdjustment(dataset->calibration(), dataset->imageSize(), {cf});

	for (int i = 0; i < T_init.size(); i++)
	{
		photoba->initial_trajectory.push_back(T_init[i], i);
		const Eigen::Isometry3d T_w(photoba->initial_trajectory.back());
		Init_traj_data_from_relativePose.push_back({T_w.matrix(), i});
		Init_traj_data.push_back({T_init_abs_pose[i], i});
	}
		EigenAlignedContainer_<Mat44> T_opt;
		cv::Mat_<float> zmap;
		UniquePointer<DatasetFrame> frame;

	// load environment light path
	std::string EnvMapPath=cf.get<std::string>("EnvMapPath");
	std::string EnvMapPosePath=cf.get<std::string>("EnvMapPosePath");

	// convert environment light pose the coordinate system of the first camera in PBA sequence
	std::vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>> trajectoryPoses;
//	string fileName = "/home/lei/Documents/Dataset/dataSetPBA/sequences/02/poses.txt";
	string fileName = "../data/dataSetPBA_init_poor/seq_17/cam_interpolated_poses_Env.txt";
//	string fileName = "/home/lei/Documents/Dataset/dataSetPBA/sequences/13/cam_interpolated_poses_Env.txt";
	// transform env light pose to the coordinate system of the first camera in PBA sequence
	readCtrlPointPoseData(fileName, trajectoryPoses);
	Sophus::SE3f frontCamPose_w (trajectoryPoses[0]);
	T_head_frame_c2w = frontCamPose_w;
	std::vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>> EnvGTPose;
	readCtrlPointPoseData(EnvMapPosePath, EnvGTPose);
	pcl::PointCloud<pcl::PointXYZ>::Ptr ControlpointCloud(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PCDWriter writer;
	int counter_controlpoint = 0;
	for (size_t i = 1; i <= EnvGTPose.size(); i++) {
		counter_controlpoint++;
		cv::Point3f pointBase = Vec3f(EnvGTPose[i - 1].translation().x(),
		                              EnvGTPose[i - 1].translation().y(),
		                              EnvGTPose[i - 1].translation().z());
		ControlpointCloud->push_back(pcl::PointXYZ(pointBase.x, pointBase.y, pointBase.z));
	}

	photoba->EnvMapPath=EnvMapPath;

	PBANL::envLightLookup  *EnvLightLookup= new PBANL::envLightLookup(argc, argv, EnvMapPath,EnvMapPosePath,frontCamPose_w);
	photoba->EnvLightLookup= EnvLightLookup;

	if (photoba->EnvLightLookup->envLightIdxMap.size() == 0)
	{
		std::cout<<"No environment light data!"<<std::endl;
		return -1;
	}
// control point cloud
	if (show_gui) {
			pangolin::CreateWindowAndBind("Main", 1800, 1000);
			glEnable(GL_DEPTH_TEST);
			// main parent display for images and 3d viewer
//			pangolin::View& main_view = pangolin::Display("main")
//		                                        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
//												.SetLayout(pangolin::LayoutEqualVertical);

			pangolin::View& main_view = pangolin::Display("main")
												.SetBounds(0.0, 1.0, 0, 1.0);


		    float heightFraction = 0.5; // 30% of the window height
		    float widthFraction = 0.5; // 30% of the window width

		    constexpr int UI_HEIGHT_img = 450; // Example height in pixels
		    constexpr int UI_WIDTH_img = 550;  // Example width in pixels
		    constexpr int RIGHT_MARGIN_img = 50; // Margin from the right boundary in pixels
		    constexpr int BOTTOM_MARGIN_img = 50; // Margin from the bottom boundary in pixels
		    pangolin::View& img_view_display = pangolin::Display("images");
		    img_view_display.SetBounds(
		            pangolin::Attach::Pix(BOTTOM_MARGIN_img),                                        // Bottom: Margin from the bottom
		            pangolin::Attach::Pix(BOTTOM_MARGIN_img + UI_HEIGHT_img),                            // Top: Bottom margin + UI_HEIGHT
		            pangolin::Attach::ReversePix(RIGHT_MARGIN_img + UI_WIDTH_img),                       // Left: Window width - (Right margin + UI_WIDTH)
		            pangolin::Attach::ReversePix(RIGHT_MARGIN_img));                       // Right: Align with the right of the main view (adjusted below)

		    main_view.AddDisplay(img_view_display);
			pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,pangolin::Attach::Pix(UI_WIDTH));

			// extra options panel
			pangolin::View& hidden_panel = pangolin::CreatePanel("hidden").SetBounds(
					0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH),
					pangolin::Attach::Pix(2 * UI_WIDTH));
			ui_show_hidden.Meta().gui_changed = true;


		    // 2D image views
		    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
		    while (img_view.size() < NUM_CAMS) {
			    cout<<"NUM_CAMS: "<<NUM_CAMS<<endl;
			    std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);
			    size_t idx = img_view.size();
			    img_view.push_back(iv);
			    img_view_display.AddDisplay(*iv);
			    iv->extern_draw_function =
			            std::bind(&draw_image_overlay, std::placeholders::_1, idx);
		    }


		    // 3D visualization (initial camera view optimized to see full map)
			pangolin::OpenGlRenderState camera(pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
					pangolin::ModelViewLookAt(-3.4, -3.7, -8.3, 2.1, 0.6, 0.2,pangolin::AxisNegY));

			pangolin::View& display3D = pangolin::Display("scene").SetAspect(-640 / 480.0).SetHandler(new pangolin::Handler3D(camera));
			main_view.AddDisplay(display3D);
			while (!pangolin::ShouldQuit()) {
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				if (ui_show_hidden.GuiChanged()) {
					hidden_panel.Show(ui_show_hidden);
					const int panel_width = ui_show_hidden ? 2 * UI_WIDTH : UI_WIDTH;
					main_view.SetBounds(0.0, 1.0, pangolin::Attach::Pix(panel_width), 1.0);
				}
				display3D.Activate(camera);

//			    if (images.find(fcid) != images.end()) {
//				    pangolin::TypedImage img = pangolin::LoadImage(images[fcid]);
//				    img_view[0]->SetImage(img);
//			    } else {
//				    img_view[0]->Clear();
//			    }




//				glClearColor(0.95f, 0.95f, 0.95f, 1.0f);// light gray background
			    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);// light gray background
				draw_scene(result, Init_traj_data,Init_traj_data_from_relativePose);
			    optimizeSignal = false;
				img_view_display.Activate();
				pangolin::FinishFrame();
				if (continue_next && ! optimizeSignal) {
							continue_next = next_step(img_view);
//				            if (optimizeSignal){ continue ;}
//				            pangolin::TypedImage img = pangolin::LoadImage(images[fcid]);
//				            img_view[0]->SetImage(img);



				} else {
							std::this_thread::sleep_for(std::chrono::milliseconds(5));
				}
			}
		}
    auto output_fn = options.get<std::string>("output");
    Info("Writing refined poses to %s\n", output_fn.c_str());
    writePosesTumRGBDFormat(output_fn, result.poses, dataset->getTimestamp());

	delete photoba;
	photoba = nullptr;

    return 0;
}

void draw_scene( PhotometricBundleAdjustment::Result & res,  EigenAlignedContainer_<CamWithId>& GTtraj_data, EigenAlignedContainer_<CamWithId> & Init_traj_data_from_relativePose)
{


	// 绘制坐标系
	glLineWidth(3);
	glBegin (GL_LINES);

	// axis x
	glColor3f ( 0.8f,0.f,0.f );
	glVertex3f( 0,0, 0 );
	glVertex3f( 0.4,0, 0 );
	// axis y
	glColor3f( 0.f,0.8f,0.f);
	glVertex3f( 0,0, 0 );
	glVertex3f( 0,0.4,0 );
	// axis z
	glColor3f( 0.2f,0.2f,1.f);
	glVertex3f( 0,0, 0 );
	glVertex3f( 0,0,0.4);

	const u_int8_t color_camera_current[3]{255, 0, 0};         // red
	const u_int8_t color_camera_left[3]{0, 125, 0};            // dark green
	const u_int8_t color_camera_right[3]{0, 0, 125};           // dark blue
	const u_int8_t color_points[3]{0, 0, 0};                   // black
	const u_int8_t color_old_points[3]{170, 170, 170};         // gray
	const u_int8_t color_selected_left[3]{0, 250, 0};          // green
	const u_int8_t color_selected_right[3]{0, 0, 250};         // blue
	const u_int8_t color_selected_both[3]{0, 250, 250};        // teal
	const u_int8_t color_outlier_observation[3]{250, 0, 250};  // purple

	if (show_cameras3d) {
		for (const auto& cam : GTtraj_data) {
			if (cam.id==0){visnav::render_camera(cam.pose.matrix(), 2.0f, color_camera_right,0.1f);}
			visnav::render_camera(cam.pose.matrix(), 3.0f, color_camera_right,0.1f);
		}

		glLineWidth(2);
		glBegin(GL_LINE_STRIP);
		glColor3f(0.0, 1.0, 0.0);
		for (size_t i = 0; i < GTtraj_data.size(); i++) {
			Eigen::Vector3d pose_translation= GTtraj_data[i].pose.block<3,1>(0,3);
			pangolin::glVertex(pose_translation);
		}
		glEnd();


		// render Init_traj_data_from_relativePose trajectory
		if (show_trajectory && Init_traj_data_from_relativePose.size() > 0) {
			glLineWidth(2);
			glBegin(GL_LINE_STRIP);
			glColor3f(1.0, 1.0, 0.0);
			for (size_t i = 0; i < Init_traj_data_from_relativePose.size(); i++) {
				// Eigen::Vector3d point = Eigen::Vector3d::Identity();
				Eigen::Vector3d pose_translation= Init_traj_data_from_relativePose[i].pose.block<3,1>(0,3);
				pangolin::glVertex(pose_translation);
			}
			glEnd();
			for(const auto& camId : Init_traj_data_from_relativePose){
					//if (camId.id==0){visnav::render_camera(camId.pose.matrix(), 1.0f, color_selected_both,0.1f);}
					//if (camId.id % 3 == 0) {
					visnav::render_camera(camId.pose.matrix(), 1.0f, color_camera_current,0.1f);
				    //}
			}
		}

		// render the control points

		if (photoba->EnvLightLookup->envLightIdxMap.size() > 0) {
			glPointSize(3.0);
			glBegin(GL_POINTS);
			for (const auto& kv_lm : res.refinedPoints) {
				    allRefinedPoints.push_back(kv_lm);
				    glColor3ubv(color_points);
				    pangolin::glVertex(kv_lm);
			}

			glEnd();
		}
		// render the refined trajectory
		if (show_trajectory && res.poses.size() > 0) {
			glLineWidth(2);
			glBegin(GL_LINE_STRIP);
			glColor3f(0.0, 0.0, 1.0);
			for (size_t i = 0; i < res.poses.size(); i++) {
			Eigen::Vector3d pose_translation= res.poses[i].block<3,1>(0,3);
			pangolin::glVertex(pose_translation);
			}
			glEnd();
			for (int i_int = 0; i_int < res.poses.size(); i_int++) {
				visnav::render_camera(res.poses[i_int].matrix(), 3.0f, color_outlier_observation,0.1f);
			}
		}

		// render control map points --------------anchor-------------------------
//		if (photoba->EnvLightLookup->envLightIdxMap.size() > 0) {
//			glPointSize(6.0);
//			glBegin(GL_POINTS);
//			for (const auto& envLight : photoba->EnvLightLookup->envLightIdxMap) {
//				Vec3 point(envLight.first.x, envLight.first.y, envLight.first.z);
//				glColor3ubv(color_camera_left);
//				pangolin::glVertex(point);
//			}
//
//			glEnd();
//		}

		// render old landmark points
		if (show_old_points3d && allRefinedPoints.size() > 0) {
			glPointSize(3.0);  // original: 3
			glBegin(GL_POINTS);

			for (const auto& kv_lm : allRefinedPoints) {
				glColor3ubv(color_old_points);  // gray
				pangolin::glVertex(kv_lm);
			}
			glEnd();
		}
	}
}
// Visualize features and related info on top of the image views
void draw_image_overlay(pangolin::View& v, size_t view_id) {
	UNUSED(v);
}

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch ( depth ) {
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
	}

	r += "C";
	r += (chans+'0');

	return r;
}


bool next_step(  std::vector<std::shared_ptr<pangolin::ImageView>>& img_view ){
	EigenAlignedContainer_<Mat44> T_opt;
	cv::Mat_<float> zmap;
	UniquePointer<DatasetFrame> frame;

	std::string dataFolder="/home/lei/Documents/Research/nldso_photometricLoss/dataAnalysis/seq_17/lvl_1_pose/";

	int lvl=0;

	photoba->lvl = lvl;
	photoba->_calib.setKforImpyramid(lvl);
	photoba->setImage_size(lvl);
	photoba->_mask.resize(photoba->_image_size.rows, photoba->_image_size.cols);
	photoba->_saliency_map.resize(photoba->_image_size.rows, photoba->_image_size.cols);
//	std::cout<<"show photoba->_calib._K_orig()\n "<<photoba->_calib._K_orig.matrix()<<std::endl;
//	std::cout<<"show new photoba->_calib.K():\n "<<photoba->_calib.K().matrix()<<std::endl;
//	std::cout<<"check _K_inv outside addFrame"<< photoba->_calib.K().matrix().inverse()<<std::endl;
//	std::cout <<"show image size: "<<photoba->_image_size.rows<<" "<<photoba->_image_size.cols<<std::endl;
	photoba->_K_inv = photoba->_calib.K().matrix().inverse().matrix();

	for(; (frame = dataset->getFrame(fid, lvl)) && !gStop; ++fid ){

		if (fid==T_init.size()-3) {

			auto output_fn = dataFolder+ "refined_poses_es_tum_abs_seq17_new_pose_NLPBAvideo"+ std::to_string(lvl)+ ".txt";
			writePosesTumRGBDFormat(output_fn, result.poses, dataset->getTimestamp());
			std::cout <<"End of dataset reached\n";
//			exit(1);
			return false;

		}
		printf("Frame %05d\n", fid);
//		cv::imshow("frame->image()",frame->image());

		cv::Mat cvImage = frame->rgbImage();
// convert cvImage from brg to rgb
//		cv::cvtColor(cvImage, cvImage, cv::COLOR_BGR2RGB);
//		cv::imshow("cvImage",cvImage);
		string ty=type2str( cvImage.type());
		cout<<"show frame->image() type:"<<ty<<endl;
		std::cout<<"show frame->image() channels:"<<frame->image().channels()<<std::endl;
//		cv::waitKey(0);
		// Determine the pixel format
		pangolin::PixelFormat fmt = pangolin::PixelFormatFromString("BGR24");
		// Create a new TypedImage with the same dimensions and pixel format
		pangolin::TypedImage img(cvImage.cols, cvImage.rows, fmt);

		// Copy the data from cv::Mat to TypedImage
		if (cvImage.isContinuous()) {
			memcpy(img.ptr, cvImage.data, cvImage.cols * cvImage.rows * cvImage.elemSize());
		} else {
			// Handle non-continuous images by copying row by row
			for (int i = 0; i < cvImage.rows; ++i) {
				memcpy(img.ptr + i * img.pitch, cvImage.ptr(i), cvImage.cols * cvImage.elemSize());
			}
		}
//		pangolin::TypedImage img = frame->image();
		img_view[0]->SetImage(img);

//		cv::waitKey(0);

		const uint8_t* I = frame->image().ptr<const uint8_t>();
		float* Z =frame->depth().ptr<float>();

		const Vec3f* N = frame->normal().ptr<Vec3f>();



//		cout<<"show frame->normal() type:"<<frame->normal().type()<<endl;
//		cv::imshow("frame->normal()",frame->normal());
//		waitKey(0)	;

		const float* R= frame->roughness().ptr<float>();

		if (N==nullptr || R==nullptr){
			std::cout<<"N or R  is nullptr"<<std::endl;
			return false;
		}

		photoba->addFrame(I, frame->image(), Z,frame->depth() ,frame->normal(), frame->roughness(),T_init[fid],  &result);

		if(optimizeSignal) {
			optimizeSignal=false;
			++fid;
			return true;
		}

		// return false before the last frame
//		if (fid==T_init.size()-10) {
//			auto output_fn = "refined_poses_es_tum_abs_pose.txt";
//			writePosesTumRGBDFormat(output_fn, result.poses, dataset->getTimestamp());
//			std::cout <<"End of dataset reached\n";
//			return false;
//		}

	}
}

