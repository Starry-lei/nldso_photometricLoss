//
// Created by lei on 15.01.23.
//

#include "ctrlPointSelector.h"


namespace DSONL {

//
    void readCtrlPointPose(string fileName, vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>> &pose) {

        ifstream trajectory(fileName);
        if (!trajectory.is_open()) {
            cout << "No controlPointPose data!" << fileName << endl;
            return;
        }

        float qw, qx, qy, qz, tx, ty, tz;
        string line;
        while (getline(trajectory, line)) {
            stringstream lineStream(line);
            lineStream >> qw >> qx >> qy >> qz >> tx >> ty >> tz;

            Eigen::Vector3f t(tx, ty, tz);
            Eigen::Quaternionf q = Eigen::Quaternionf(qw, qx, qy, qz).normalized();
            Sophus::SE3f SE3_qt(q, t);
            pose.push_back(SE3_qt);
        }

    }


    DSONL::ctrlPointSelector::ctrlPointSelector(Sophus::SE3d Camera1_extrin, string controlPointPose_path, Mat Image,
                                                Mat depthImage,
                                                Eigen::Matrix<float, 3, 3> &K,
    , Mat pointOfInterest) {

        // constants
        kNearest = 1;
        Sophus::SE3f Camera1_w2c = Camera1_extrin.cast<float>();
        float fx = K(0, 0), cx = K(0, 2), fy = K(1, 1), cy = K(1, 2);


        Vec2i boundingBoxUpperLeft( 83, 76);// 83, 76
        Vec2i boundingBoxBotRight(240, 320);





        pcl::PCDWriter writer;

        // check PointCloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr ControlpointCloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr nearestPointCloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr scenePointCloud(new pcl::PointCloud<pcl::PointXYZ>());

        // variables
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        std::unordered_map<cv::Point3f, vector<int>, hash3d<cv::Point3f>, equalTo<cv::Point3f>> envLightMap;
        std::vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>> controlPointPoses;
        readCtrlPointPose(controlPointPose_path, controlPointPoses);
        std::cout << " \n Show controlPointPoses size:" << controlPointPoses.size() << std::endl;
        std::unordered_map<cv::Point3f, int, hash3d<cv::Point3f>, equalTo<cv::Point3f>> pointCloud_UnorderedMap;


        for (size_t i = 1; i <= controlPointPoses.size(); i++) {
            Sophus::SE3f envMapPose_world = controlPointPoses[i - 1].cast<float>();
            cv::Point3f pointBase = Vec3f(controlPointPoses[i - 1].translation().x(),
                                          controlPointPoses[i - 1].translation().y(),
                                          controlPointPoses[i - 1].translation().z());
            ControlpointCloud->push_back(pcl::PointXYZ(pointBase.x, pointBase.y, pointBase.z));
            pointCloud_UnorderedMap.insert(make_pair(pointBase, (int) i));
        }

        if (ControlpointCloud->empty()) { std::cerr << "\n Wrong Control-pointCloud!" << endl; }
        kdtree.setInputCloud(ControlpointCloud);
        imshow("slamImg", Image);

        std::unordered_map<int, int> inliers_filter, inliers_filter_i;


//        inliers_filter.emplace(108, 97 );//cabinet
//        inliers_filter.emplace(125, 102);//table
        // new test point in shadow:  108, 108
        inliers_filter.emplace( 112, 130); // 112, 130

        Mat checkingArea(depthImage.rows, depthImage.cols, CV_64FC1, Scalar(0));
        for (int u = 0; u < depthImage.rows; u++)// colId, cols: 0 to 480
        {
            for (int v = 0; v < depthImage.cols; v++)// rowId,  rows: 0 to 640
            {

//                if(inliers_filter.count(u)==0){continue;} // ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//                if(inliers_filter[u]!=v ){continue;} // ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~



                //  use bounding box here
//                if ( (v<boundingBoxUpperLeft.val[1] || v>boundingBoxBotRight.val[1]) || (u< boundingBoxUpperLeft.val[0] ||  u> boundingBoxBotRight.val[0])){ continue;}

                if (pointOfInterest.at<uchar>(u,v)!=255){ continue;}

                // marks
                checkingArea.at<double>(u, v)= Image.at<double>(u, v);


                // projection
                double iDepth = depthImage.at<double>(u, v);
                Eigen::Vector2f pixelCoord((float) v, (float) u);//  u is the row id , v is col id
                Eigen::Vector3f p_3d_no_d((pixelCoord(0) - cx) / fx, (pixelCoord(1) - cy) / fy, (float) 1.0);
                Eigen::Vector3f p_c1;
                p_c1 = (float) iDepth * p_3d_no_d;

                // convert it to world coordinate system
                Eigen::Vector3f p_w1 = Camera1_w2c * p_c1;
                pcl::PointXYZ searchPoint(p_w1.x(), p_w1.y(), p_w1.z());

                // add into point cloud (world coordinate)
                 scenePointCloud->push_back(searchPoint);

//                cv::Point3f key_shaderPoint(p_c1.x(), p_c1.y(), p_c1.z());

                std::vector<int> pointIdxKNNSearch(kNearest);
                std::vector<float> pointKNNSquaredDistance(kNearest);

//                vector<int> controlPointIndex[3];//  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! equal to kNearest(change together) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                Vec3f key4Search;
                if (kdtree.nearestKSearch(searchPoint, kNearest, pointIdxKNNSearch, pointKNNSquaredDistance) > 0) {


                std::cout << "\n---The Nearest point---"<<
                                    (*(ControlpointCloud))[ pointIdxKNNSearch[0]].x
                          << " " << (*(ControlpointCloud))[ pointIdxKNNSearch[0]].y
                          << " " << (*(ControlpointCloud))[ pointIdxKNNSearch[0]].z
                          << " (squared distance: " << pointKNNSquaredDistance[0] << ")" << std::endl;

//                    cout<<"\n Show current shader point:\n"<<p_w1<<"\n show nearst envMap point coordinate:\n"<< (*(ControlpointCloud))[ pointIdxKNNSearch[0]]<<endl;

//                    cout<<"show pointIdxKNNSearch.size()"<<pointIdxKNNSearch.size()<<endl;

                    nearestPointCloud->push_back(pcl::PointXYZ((*(ControlpointCloud))[pointIdxKNNSearch[0]].x,
                                                                   (*(ControlpointCloud))[pointIdxKNNSearch[0]].y,
                                                                   (*(ControlpointCloud))[pointIdxKNNSearch[0]].z));

                    for (std::size_t idx = 0; idx < pointIdxKNNSearch.size(); ++idx) {
                        Vec3f key4Search_;
                        key4Search_.val[0] = (*(ControlpointCloud))[pointIdxKNNSearch[idx]].x;
                        key4Search_.val[1] = (*(ControlpointCloud))[pointIdxKNNSearch[idx]].y;
                        key4Search_.val[2] = (*(ControlpointCloud))[pointIdxKNNSearch[idx]].z;
//                        cout<<"show key4Search_"<<key4Search_<<endl;
//                        controlPointIndex->push_back(pointCloud_UnorderedMap[key4Search_]);

//                        cout<<"show pointCloud_UnorderedMap[key4Search_]"<< pointCloud_UnorderedMap[key4Search_]<<endl;
                        selectedIndex_vec.push_back(pointCloud_UnorderedMap[key4Search_]);
                    }
//                    envLightMap.emplace(key_shaderPoint, *controlPointIndex);
                }
            }
        }


//        for (auto pair: envLightMap) {
//            cout << "\n show correspondence between shader point and its ctrlPointIndex:" << endl;
//            for (int i = 0; i < pair.second.size(); ++i) {
//                cout << pair.second[i] << ",";
//            }
//        }
//        writer.write("ControlpointCloud.pcd", *ControlpointCloud, false);// do we need the sensor acquisition origin?
//        writer.write("nearestPointCloud.pcd", *nearestPointCloud, false);// do we need the sensor acquisition origin?
//        writer.write("scenePointCloud.pcd", *scenePointCloud, false);// do we need the sensor acquisition origin?




        sort(selectedIndex_vec.begin(), selectedIndex_vec.end());
        selectedIndex_vec.erase(unique(selectedIndex_vec.begin(), selectedIndex_vec.end()), selectedIndex_vec.end());


        for (int idx:selectedIndex_vec) { selectedIndex.emplace(idx,1); cout<<"show selected index:"<< idx<<endl;}




        // Only one envMap here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
//                selectedIndex.emplace(int(216),1);
//                selectedIndex.emplace(int(36), 1);
//                selectedIndex.emplace(int(200),1);
//                selectedIndex.emplace(int(40),1);
////                 selectedIndex.emplace(int(63),1);
//
//                selectedIndex.emplace(int(13),1);

        //        cout<<"show selectedIndex "<< selectedIndex_vec[0]<<endl;


        cout << "\n show number of selected ctrlPoints size:" << selectedIndex_vec.size() << endl;

        imshow("checkingArea", checkingArea);
        waitKey(0);


    }

    DSONL::ctrlPointSelector::~ctrlPointSelector() {

    }


}


