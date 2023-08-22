//
// Created by lei on 05.05.23.
//

#include "envLightPreProcessing.h"

#include <tbb/parallel_for.h>
#include <vector>
#include <mutex>

using namespace std;

using namespace gsn;


namespace PBANL {


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

    void readCtrlPointPoseData(string fileName, vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>>& pose,Sophus::SE3f frontCamPose) {

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

            // convert envLightPose to frontCamPose
            SE3_qt = frontCamPose.inverse() * SE3_qt;

            pose.push_back(SE3_qt);
        }

    }

    envLightLookup::envLightLookup( int argc, char **argv,
                                   string envMap_Folder, string controlPointPose_path,Sophus::SE3f frontCamPose) {


        pcl::PCDWriter writer;
        std::vector<string> fileNames;
        GetFileNames(envMap_Folder,fileNames);
        std::cout<<" \n Show fileNames.size():"<<fileNames.size()<< std::endl;

        std::vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>> controlPointPoses;
        readCtrlPointPoseData(controlPointPose_path,controlPointPoses, frontCamPose);
        std::cout<<" \n Show controlPointPoses.size():"<<controlPointPoses.size()<< std::endl;

//        assert(fileNames.size()=controlPointPoses.size())

        // save all the translation of control points as pcl point cloud
//        pcl::PointCloud<pcl::PointXYZ>::Ptr ControlpointCloud (new pcl::PointCloud<pcl::PointXYZ>());
//        std::map<Vec3f, int> envLightIdxMap;



        //brdf(Red&Green Table)
        string brdfIntegrationMap_path = "../include/brdfIntegrationMap/brdfIntegrationMap.pfm";
        DSONL::brdfIntegrationMap *brdfIntegrationMap= new DSONL::brdfIntegrationMap(brdfIntegrationMap_path);
        brdfIntegrationMap->makebrdfIntegrationMap(brdfSampler);
        delete brdfIntegrationMap;
        ControlpointCloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
        std::mutex mtx;
        int counter=0;
        string renderedEnvLight_path= envMap_Folder;

        tbb::parallel_for(
                (size_t)1, (size_t)(fileNames.size()+1), [&] (size_t i) {

//                    if (selectedIndex.count(static_cast<int>(i)) == 0 ) { return ; }
                    pointEnvlight pEnv;
                    pEnv.ctrlPointIdx = static_cast<int>(i);
                    pEnv.envMapPose_world = controlPointPoses[i - 1].cast<float>();
                    pEnv.pointBase = Vec3f(controlPointPoses[i - 1].translation().x(),
                                           controlPointPoses[i - 1].translation().y(),
                                           controlPointPoses[i - 1].translation().z());

                    {
                        std::lock_guard<std::mutex>grd(mtx);
                        ControlpointCloud->push_back(pcl::PointXYZ(pEnv.pointBase.x, pEnv.pointBase.y, pEnv.pointBase.z));
                        envLightIdxMap.insert(make_pair(pEnv.pointBase, static_cast<int>(i)));
                        counter += 1;
//                        cout << "\n show current envMap index: " << i << endl;
//                        cout << "\n show number of envMap added: " << counter << endl;
                    }
                });

        // sparsify the control point cloud

        ofstream sparsifyPointCloud;
        sparsifyPointCloud.open ("ControlpointCloud_Sparsfied.txt");
        pcl::PointCloud<pcl::PointXYZ>::Ptr ControlpointCloud_voxelgrid (new pcl::PointCloud<pcl::PointXYZ>);

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_temp;
        kdtree_temp.setInputCloud(ControlpointCloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr downSampledCloud (new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<int> pointIdxKNNSearch(1);
        std::vector<float> pointKNNSquaredDistance(1);

        if (false){
            downSampledCloud->clear();
            // save point cloud
            writer.write("ControlpointCloud_not_sparsfied.pcd",*ControlpointCloud, false);
            std::cout << "EnvMap PointCloud before filtering: " << ControlpointCloud->width * ControlpointCloud->height<< " data points (" << pcl::getFieldsList (*ControlpointCloud) << ")." << std::endl;
            pcl::VoxelGrid<pcl::PointXYZ>  downsample;
            downsample.setInputCloud(ControlpointCloud);
            downsample.setLeafSize(0.1f,0.1,0.1f); // leaf size: 1cm, 5m
            downsample.filter(*ControlpointCloud_voxelgrid);

//            downsample.setFilterLimits()

            std::vector<int> neighbor_indices;
            for (const auto& point: *ControlpointCloud_voxelgrid){
                sparsifyPointCloud <<point.x<<" "<<point.y<<" "<<point.z<<"\n";
                pcl::PointXYZ searchPoint(point.x,point.y,point.z);
                if ( kdtree_temp.nearestKSearch (searchPoint, 1, pointIdxKNNSearch, pointKNNSquaredDistance) > 0 ){

//                    std::cout << "    "  <<   (*ControlpointCloud)[ pointIdxKNNSearch[0] ].x
//                              << " " << (*ControlpointCloud)[ pointIdxKNNSearch[0] ].y
//                              << " " << (*ControlpointCloud)[ pointIdxKNNSearch[0] ].z
//                              << " (squared distance: " << pointKNNSquaredDistance[0] << ")" << std::endl;

                    downSampledCloud->push_back(pcl::PointXYZ( (*ControlpointCloud)[ pointIdxKNNSearch[0] ].x, (*ControlpointCloud)[ pointIdxKNNSearch[0] ].y, (*ControlpointCloud)[ pointIdxKNNSearch[0] ].z));

                }
            }
            // save sparsified point cloud
            writer.write("ControlpointCloud_sparsfied.pcd",*ControlpointCloud_voxelgrid, false);
        }

        sparsifyPointCloud.close();

        if (ControlpointCloud->empty()){std::cerr<<"\n Wrong Control-pointCloud!"<< endl;}
        kdtree.setInputCloud(ControlpointCloud);


        // save ControlpointCloud
        writer.write("ControlpointCloud_afterTransformation.pcd",*ControlpointCloud, false);

        // ======================serialization============================
    }

    pointEnvlight::pointEnvlight(const pointEnvlight & old) {
        pointEnvlight();
        envMapPose_world=old.envMapPose_world;
        pointBase= old.pointBase;
        EnvmapSampler.push_back(old.EnvmapSampler[0]);
        EnvmapSampler.push_back(old.EnvmapSampler[1]);
        ctrlPointIdx= old.ctrlPointIdx ;
    }


}