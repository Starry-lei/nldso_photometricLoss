//
// Created by lei on 04.05.23.
//

#ifndef NLDSO_PHOTOMETRICLOSS_PANGOLINVIS_H
#define NLDSO_PHOTOMETRICLOSS_PANGOLINVIS_H


#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>
#include <Eigen/Dense>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/tbb.h>
#include <unordered_map>
#include <sophus/se3.hpp>





/// identifies a frame of multiple images (stereo pair)
using FrameId = int64_t;

/// identifies the camera (left or right)
using CamId = std::size_t;

/// Ids for feature tracks; also used for landmarks created from (some of) the
/// tracks;
using TrackId = int64_t;

constexpr int PIXELS_IN_PATCH = 8;
using Button = pangolin::Var<std::function<void(void)>>;
/// pair of image timestamp and camera id identifies an image (imageId)
typedef std::pair<FrameId, CamId> TimeCamId;
std::ostream &operator<<(std::ostream &os, const TimeCamId &tcid) {
    os << tcid.first << "_" << tcid.second;
    return os;
}
/// contains info about patches around a keypoint for the photometric error
struct Patch {
    Eigen::Matrix<double, 2, PIXELS_IN_PATCH> positions;
    Eigen::Matrix<double, PIXELS_IN_PATCH, 1> intensities;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// ids for 2D features detected in images
using FeatureId = int;

/// Feature tracks are collections of {ImageId => FeatureId}.
/// I.e. a collection of all images that observed this feature and the
/// corresponding feature index in that image.
using FeatureTrack = std::map<TimeCamId, FeatureId>;

//struct Calibration {
//    typedef std::shared_ptr<Calibration> Ptr;
//
//    Calibration() {}
//
//    // transformations from cameras to body (IMU)
//    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> T_i_c;
//
//    // Camera intrinsics
//    Eigen::Matrix<double,3,3> intrinsics;
//};



/// photometric landmarks in the map
struct PhotoLandmark {
    /// If selected is true, this photolandmark was not born from an old ORB
    /// feature but rather selected due to its high gradient
    bool selected;

    /// Patch
    Patch patch;

    /// Inverse distance to host frame
    double d;

    /// Host Frame
    std::pair<TimeCamId, FeatureId> host;

    /// Map of frames observing the landmark
    FeatureTrack obs;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


using PhotoLandmarks = std::unordered_map<
        TrackId, PhotoLandmark, std::hash<TrackId>, std::equal_to<TrackId>,
        Eigen::aligned_allocator<std::pair<const TrackId, PhotoLandmark>>>;

///////////////////////////////////////////////////////////////////////////////
/// point positions for an image (selected and matched)
struct PhotoCandidatePointsData {
    /// Points selected due to their high gradient (reside in their host frame)
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
            selected_points;
    /// Points matched to a selected points (reside in an "observation" image)
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
            matched_points;
};

using PhotoCandidatePoints =
        tbb::concurrent_unordered_map<TimeCamId, PhotoCandidatePointsData>;

///////////////////////////////////////////////////////////////////////////////

/// cameras in the map
struct Camera {
    // camera pose (transforms from camera to world)
    Sophus::SE3d T_w_c;

    // affine transfer function parameters // 'a' and 'b'
    std::vector<double> affine_ab;

    double max_inv_distance, min_inv_distance;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// collection {imageId => Camera} for all cameras in the map
using Cameras =std::map<CamId, Camera, std::less<CamId>,
                Eigen::aligned_allocator<std::pair<const CamId, Camera>>>;

// create a class for visualization
class PangolinVis
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

//public:
//    PangolinVis();
//    ~PangolinVis();
//    static constexpr int UI_WIDTH = 200;
//    static constexpr int NUM_CAMS = 2;

};

void render_camera(const Eigen::Matrix4d& T_w_c, float lineWidth,
                   const u_int8_t* color, float sizeFactor) {
    glPushMatrix();
    glMultMatrixd(T_w_c.data());
    glColor3ubv(color);
    glLineWidth(lineWidth);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    const float sz = sizeFactor;
    const float width = 640, height = 480, fx = 500, fy = 500, cx = 320,
            cy = 240;  // choose an arbitrary intrinsics because we don't need
    // the camera be exactly same as the original one
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
    glEnd();
    glPopMatrix();
}






#endif //NLDSO_PHOTOMETRICLOSS_PANGOLINVIS_H















//note



///////////////////////////////////////////////////////////////////////////////
/// Declarations
///////////////////////////////////////////////////////////////////////////////

void draw_image_overlay(pangolin::View& v, size_t cam_id);
void draw_scene();
void optimize_photo();

void pangolinVis();
void setup_pangolin();

///////////////////////////////////////////////////////////////////////////////
/// Constants
///////////////////////////////////////////////////////////////////////////////

constexpr int UI_WIDTH = 200;
constexpr int NUM_CAMS = 2;


///////////////////////////////////////////////////////////////////////////////
/// GUI parameters
///////////////////////////////////////////////////////////////////////////////

// The following GUI elements can be enabled / disabled from the main panel by
// switching the prefix from "ui" to "hidden" or vice verca. This way you can
// show only the elements you need / want for development.

pangolin::Var<bool> ui_show_hidden("ui.show_extra_options", false, false, true);

//////////////////////////////////////////////
/// Image display options

pangolin::Var<int> show_frame1("ui.show_frame1", 0, 0, 1500);
pangolin::Var<int> show_cam1("ui.show_cam1", 0, 0, NUM_CAMS - 1);
pangolin::Var<int> show_frame2("ui.show_frame2", 0, 0, 1500);
pangolin::Var<int> show_cam2("ui.show_cam2", 1, 0, NUM_CAMS - 1);

pangolin::Var<bool> show_cameras3d("hidden.show_cameras", false, false, true);
pangolin::Var<bool> show_gt_cameras3d("hidden.show_gt_cameras", true, false,
                                      true);
pangolin::Var<bool> show_geometric_cameras3d("hidden.show_geometric_cameras",
                                             true, false, true);
pangolin::Var<bool> show_points3d("hidden.show_points", true, false, true);

// Photometric part
pangolin::Var<bool> show_selected("ui.show_selected", true, false, true);
pangolin::Var<bool> show_photomatches("ui.show_photomatches", true, false,true);


/// Candidate Points
PhotoCandidatePoints photo_candidates_host_frames;
/// photometric landmark positions and feature observations in current map
PhotoLandmarks photolandmarks;
/// camera poses in the current map
Cameras cameras;
/// camera poses after the geometric BA
Cameras geometric_cameras;

/// groundtruth camera poses in the current map
Cameras gt_cameras;

/// intrinsic calibration
Calibration calib_cam;

std::atomic<bool> opt_running{false};
std::atomic<bool> opt_finished{false};

std::shared_ptr<std::thread> opt_thread;


// Visualize features and related info on top of the image views
void draw_image_overlay(pangolin::View &v, size_t view_id) {
    size_t frame_id = view_id == 0 ? show_frame1 : show_frame2;
    size_t cam_id = view_id == 0 ? show_cam1 : show_cam2;

    TimeCamId tcid = std::make_pair(frame_id, cam_id);

    float text_row = 20;
    if (show_selected) {
        glLineWidth(1.0);
        glColor3f(1.0, 0.0, 0.0);  // red
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        if (photo_candidates_host_frames.find(tcid) !=
            photo_candidates_host_frames.end()) {
            const PhotoCandidatePointsData &cr =
                    photo_candidates_host_frames.at(tcid);

            for (size_t i = 0; i < cr.selected_points.size(); i++) {
                Eigen::Vector2d c = cr.selected_points[i];
                // double angle = cr.corner_angles[i];
                pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

                //        Eigen::Vector2d r(3, 0);
                //        Eigen::Rotation2Dd rot(angle);
                //        r = rot * r;
                //        pangolin::glDrawLine(c, c + r);
            }

            pangolin::GlFont::I()
                    .Text("Selected %d points", cr.selected_points.size())
                    .Draw(5, text_row);

        } else {
            glLineWidth(1.0);

            pangolin::GlFont::I().Text("No candidates selected").Draw(5, text_row);
        }
        text_row += 20;
    }

    if (show_photomatches) {
        glLineWidth(1.0);
        glColor3f(0.0, 0.0, 1.0);  // blue
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        if (photo_candidates_host_frames.find(tcid) !=
            photo_candidates_host_frames.end()) {
            const PhotoCandidatePointsData &cr =
                    photo_candidates_host_frames.at(tcid);

            for (size_t i = 0; i < cr.matched_points.size(); i++) {
                Eigen::Vector2d c = cr.matched_points[i];
                // double angle = cr.corner_angles[i];
                pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

                //        Eigen::Vector2d r(3, 0);
                //        Eigen::Rotation2Dd rot(angle);
                //        r = rot * r;
                //        pangolin::glDrawLine(c, c + r);
            }

            pangolin::GlFont::I()
                    .Text("Selected %d points", cr.matched_points.size())
                    .Draw(5, text_row);

        } else {
            glLineWidth(1.0);

            pangolin::GlFont::I().Text("No candidates selected").Draw(5, text_row);
        }
        text_row += 20;
    }
}

// Render the 3D viewer scene of cameras and points
void draw_scene() {

    const CamId tcid0(0);
    show_cameras3d = true;

    if (cameras.empty()){
        // Set  camera pose to identity
        Eigen::Matrix3d R_identity = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_zero = Eigen::Vector3d::Zero();
        cameras[tcid0].T_w_c = Sophus::SE3d(R_identity, t_zero);

        return;
    }



    const CamId tcid1 = show_cam1;


    const u_int8_t color_camera_left[3]{0, 0, 200};        // dark blue
    const u_int8_t color_camera_right[3]{0, 0, 200};       // dark blue
    const u_int8_t color_geom_camera_left[3]{200, 0, 0};   // dark red
    const u_int8_t color_geom_camera_right[3]{200, 0, 0};  // dark red
    const u_int8_t color_gt_camera_left[3]{0, 200, 0};     // dark green
    const u_int8_t color_gt_camera_right[3]{0, 200, 0};    // dark green

    const u_int8_t color_selected_left[3]{0, 0, 250};   // blue
    const u_int8_t color_selected_right[3]{0, 0, 250};  // blue

    const u_int8_t color_selected_both[3]{0, 250, 250};        // teal
    const u_int8_t color_points[3]{0, 0, 0};                   // black
    const u_int8_t color_points_selected[3]{150, 0, 0};        // red
    const u_int8_t color_outlier_points[3]{250, 0, 0};         // red
    const u_int8_t color_outlier_observation[3]{250, 0, 250};  // purple

    // render cameras
    if (show_cameras3d) {
        for (const auto &cam : cameras) {
            if (cam.first == tcid1) {
                render_camera(cam.second.T_w_c.matrix(), 3.0f, color_selected_left,
                              0.1f);
            }
//            else if (cam.first.second == 0) {
//                render_camera(cam.second.T_w_c.matrix(), 2.0f, color_camera_left, 0.1f);
//            }
        }
    }

//    // render gt cameras
//    if (show_gt_cameras3d) {
//        for (const auto &gt_cam : gt_cameras) {
//            if (gt_cam.first.second == 0) {
//                render_camera(gt_cam.second.T_w_c.matrix(), 2.0f, color_gt_camera_left,
//                              0.1f);
//            } else {
//                render_camera(gt_cam.second.T_w_c.matrix(), 2.0f, color_gt_camera_right,
//                              0.1f);
//            }
//        }
//    }
//
//    // render cameras computed from the geometric BA
//    if (show_geometric_cameras3d) {
//        for (const auto &cam : geometric_cameras) {
//            if (cam.first.second == 0) {
//                render_camera(cam.second.T_w_c.matrix(), 2.0f, color_geom_camera_left,
//                              0.1f);
//            } else {
//                render_camera(cam.second.T_w_c.matrix(), 2.0f, color_geom_camera_right,
//                              0.1f);
//            }
//        }
//    }

    // render points
    if (show_points3d && photolandmarks.size() > 0) {
        glPointSize(3.0);
        glBegin(GL_POINTS);
        for (const auto &kv_photolm : photolandmarks) {
            const TrackId track_id = kv_photolm.first;
            const PhotoLandmark &photolm = kv_photolm.second;
            CamId tcid = photolm.host.first.second;
            const Sophus::SE3d &T_w_c = cameras[tcid].T_w_c;

            if (photolm.selected)
                glColor3ubv(color_points_selected);
            else
                glColor3ubv(color_points);

//            const Eigen::Matrix<double, 3, PIXELS_IN_PATCH> P_i_3d =
//                    calib_cam.intrinsics[tcid.second]->unproject_many(photolm.patch.positions) /
//                    photolm.d;

//            for (size_t p = 0; p < PIXELS_IN_PATCH; ++p) {
//                pangolin::glVertex(T_w_c * P_i_3d.col(p));
//            }
        }
        glEnd();
    }
}

void setup_pangolin(){
    pangolin::CreateWindowAndBind("Main", 1800, 1000);
    glEnable(GL_DEPTH_TEST);

}

void pangolinVis(){


    // main parent display for images and 3d viewer
    pangolin::View& main_view =pangolin::Display("non-Lambertian PBA")
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
            .SetLayout(pangolin::LayoutEqualVertical);

    // parent display for images
    pangolin::View& img_view_display =pangolin::Display("images").SetLayout(pangolin::LayoutEqual);
    main_view.AddDisplay(img_view_display);

    // main ui panel
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    // extra options panel
    pangolin::View& hidden_panel = pangolin::CreatePanel("hidden").SetBounds(
            0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH),
            pangolin::Attach::Pix(2 * UI_WIDTH));
    ui_show_hidden.Meta().gui_changed = true;

    // 2D image views
    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
    while (img_view.size() < NUM_CAMS) {
        std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

        size_t idx = img_view.size();
        img_view.push_back(iv);

        img_view_display.AddDisplay(*iv);
        iv->extern_draw_function =
                std::bind(&draw_image_overlay, std::placeholders::_1, idx);
    }

    // 3D visualization (initial camera view optimized to see full map)
    pangolin::OpenGlRenderState camera(
            pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
            pangolin::ModelViewLookAt(-3.4, -3.7, -8.3, 2.1, 0.6, 0.2,
                                      pangolin::AxisNegY));

    pangolin::View& display3D =
            pangolin::Display("scene")
                    .SetAspect(-640 / 480.0)
                    .SetHandler(new pangolin::Handler3D(camera));
    main_view.AddDisplay(display3D);

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (ui_show_hidden.GuiChanged()) {
            hidden_panel.Show(ui_show_hidden);
            const int panel_width = ui_show_hidden ? 2 * UI_WIDTH : UI_WIDTH;
            main_view.SetBounds(0.0, 1.0, pangolin::Attach::Pix(panel_width), 1.0);
        }
        display3D.Activate(camera);
        glClearColor(0.95f, 0.95f, 0.95f, 1.0f);  // light gray background
        draw_scene();
        img_view_display.Activate();
        pangolin::FinishFrame();

    }

}


