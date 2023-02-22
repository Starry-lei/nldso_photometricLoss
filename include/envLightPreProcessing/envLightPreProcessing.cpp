//
// Created by lei on 09.01.23.
//


#include "envLightPreProcessing.h"
#include <tbb/parallel_for.h>
#include <vector>
#include <mutex>

using namespace std;


namespace DSONL {




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



    envLight::envLight(std::unordered_map<int, int> selectedIndex, int argc, char **argv, string envMap_Folder, string controlPointPose_path) {


        pcl::PCDWriter writer;



        std::vector<string> fileNames;
        GetFileNames(envMap_Folder,fileNames);
        std::cout<<" \n Show fileNames.size():"<<fileNames.size()<< std::endl;

        std::vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>> controlPointPoses;
        readCtrlPointPoseData(controlPointPose_path,controlPointPoses);
        std::cout<<" \n Show controlPointPoses.size():"<<controlPointPoses.size()<< std::endl;


        //brdf(Red&Green Table)
        string brdfIntegrationMap_path = "../include/brdfIntegrationMap/brdfIntegrationMap.pfm";
        brdfIntegrationMap *brdfIntegrationMap= new DSONL::brdfIntegrationMap(brdfIntegrationMap_path);
        brdfIntegrationMap->makebrdfIntegrationMap(brdfSampler);
        delete brdfIntegrationMap;


        ControlpointCloud.reset(new pcl::PointCloud<pcl::PointXYZ>());


        std::mutex mtx;

        int counter=0;

        int seletcEnvMap=36;
        int seletcEnvMap2=200;

//        string renderedEnvLight_path="/home/lei/Documents/Research/envMapData/renderedEnvMap";
        string renderedEnvLight_path= envMap_Folder;


//        for (size_t i=1;i<=fileNames.size();i++) {

        tbb::parallel_for(
                (size_t)1, (size_t)(fileNames.size()+1), [&] (size_t i) {

                    if (selectedIndex.count(static_cast<int>(i)) == 0) { return ; }


                    stringstream ss;
                    string img_idx_str;
                    ss << i;
                    ss >> img_idx_str;
                    string name_prefix = "/envMap";


                    string renderedEnvLightfolder =
                            renderedEnvLight_path + "/envMap" + img_idx_str + "/renderedEnvLight";
                    string renderedEnvLightDiffuse =
                            renderedEnvLight_path + "/envMap" + img_idx_str + "/renderedEnvLightDiffuse";
                    string envMapDiffuse = renderedEnvLightDiffuse + "/envMapDiffuse_" + img_idx_str + ".pfm";


                    pointEnvlight pEnv;
                    pEnv.ctrlPointIdx = static_cast<int>(i);

                    pEnv.envMapPose_world = controlPointPoses[i - 1].cast<float>();
                    pEnv.pointBase = Vec3f(controlPointPoses[i - 1].translation().x(),
                                           controlPointPoses[i - 1].translation().y(),
                                           controlPointPoses[i - 1].translation().z());


                    EnvMapLookup *EnvMapLookup = new DSONL::EnvMapLookup();
                    EnvMapLookup->makeMipMap(pEnv.EnvmapSampler,
                                             renderedEnvLightfolder); // index_0: prefiltered Env light
                    delete EnvMapLookup;

                    diffuseMap *diffuseMap = new DSONL::diffuseMap;
                    diffuseMap->makeDiffuseMap(pEnv.EnvmapSampler, envMapDiffuse); // index_1: diffuse
                    delete diffuseMap;
//                    cout << "\n processing: " << i << endl;
//                    fflush(stdout);

                    {
                        std::lock_guard<std::mutex>grd(mtx);
                        ControlpointCloud->push_back(pcl::PointXYZ(pEnv.pointBase.x, pEnv.pointBase.y, pEnv.pointBase.z));
                        envLightMap.insert(make_pair(pEnv.pointBase, pEnv));
                        counter += 1;
                        cout << "\n show current envMap index: " << i << endl;
                        cout << "\n show number of envMap added: " << counter << endl;
                    }
//    }

                }

                );

        if (ControlpointCloud->empty()){std::cerr<<"\n Wrong Control-pointCloud!"<< endl;}
        kdtree.setInputCloud(ControlpointCloud);

        // ======================serialization============================

    }

    envLightLookup::envLightLookup(std::unordered_map<int, int> selectedIndex, int argc, char **argv,
                                   string envMap_Folder, string controlPointPose_path) {


        pcl::PCDWriter writer;
        std::vector<string> fileNames;
        GetFileNames(envMap_Folder,fileNames);
        std::cout<<" \n Show fileNames.size():"<<fileNames.size()<< std::endl;

        std::vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>> controlPointPoses;
        readCtrlPointPoseData(controlPointPose_path,controlPointPoses);
        std::cout<<" \n Show controlPointPoses.size():"<<controlPointPoses.size()<< std::endl;


        //brdf(Red&Green Table)
        string brdfIntegrationMap_path = "../include/brdfIntegrationMap/brdfIntegrationMap.pfm";
        brdfIntegrationMap *brdfIntegrationMap= new DSONL::brdfIntegrationMap(brdfIntegrationMap_path);
        brdfIntegrationMap->makebrdfIntegrationMap(brdfSampler);
        delete brdfIntegrationMap;


        ControlpointCloud.reset(new pcl::PointCloud<pcl::PointXYZ>());


        std::mutex mtx;
        int counter=0;
        int seletcEnvMap=36;
        int seletcEnvMap2=200;

//        string renderedEnvLight_path="/home/lei/Documents/Research/envMapData/renderedEnvMap";
        string renderedEnvLight_path= envMap_Folder;


//        for (size_t i=1;i<=fileNames.size();i++) {

        tbb::parallel_for(
                (size_t)1, (size_t)(fileNames.size()+1), [&] (size_t i) {

                    if (selectedIndex.count(static_cast<int>(i)) == 0 ) { return ; }


//                    stringstream ss;
//                    string img_idx_str;
//                    ss << i;
//                    ss >> img_idx_str;
//                    string name_prefix = "/envMap";
//
//
//                    string renderedEnvLightfolder =
//                            renderedEnvLight_path + "/envMap" + img_idx_str + "/renderedEnvLight";
//                    string renderedEnvLightDiffuse =
//                            renderedEnvLight_path + "/envMap" + img_idx_str + "/renderedEnvLightDiffuse";
//                    string envMapDiffuse = renderedEnvLightDiffuse + "/envMapDiffuse_" + img_idx_str + ".pfm";


                    pointEnvlight pEnv;
                    pEnv.ctrlPointIdx = static_cast<int>(i);

                    pEnv.envMapPose_world = controlPointPoses[i - 1].cast<float>();
                    pEnv.pointBase = Vec3f(controlPointPoses[i - 1].translation().x(),
                                           controlPointPoses[i - 1].translation().y(),
                                           controlPointPoses[i - 1].translation().z());


//                    EnvMapLookup *EnvMapLookup = new DSONL::EnvMapLookup();
//                    EnvMapLookup->makeMipMap(pEnv.EnvmapSampler,
//                                             renderedEnvLightfolder); // index_0: prefiltered Env light
//                    delete EnvMapLookup;
//
//                    diffuseMap *diffuseMap = new DSONL::diffuseMap;
//                    diffuseMap->makeDiffuseMap(pEnv.EnvmapSampler, envMapDiffuse); // index_1: diffuse
//                    delete diffuseMap;
//                    cout << "\n processing: " << i << endl;
//                    fflush(stdout);

                    {
                        std::lock_guard<std::mutex>grd(mtx);
                        ControlpointCloud->push_back(pcl::PointXYZ(pEnv.pointBase.x, pEnv.pointBase.y, pEnv.pointBase.z));
                        envLightIdxMap.insert(make_pair(pEnv.pointBase, static_cast<int>(i)));
                        counter += 1;
                        cout << "\n show current envMap index: " << i << endl;
                        cout << "\n show number of envMap added: " << counter << endl;
                    }


//    }

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

//        *downSampledCloud=*ControlpointCloud;


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
//        std::cout << "EnvMap PointCloud after filtering: " << ControlpointCloud->width * ControlpointCloud->height<< " data points (" << pcl::getFieldsList (*downSampledCloud) << ")." << std::endl;
//        writer.write("ControlpointCloud_downSampledCloud.pcd",*ControlpointCloud, false);
//        *ControlpointCloud=*downSampledCloud;
        if (ControlpointCloud->empty()){std::cerr<<"\n Wrong Control-pointCloud!"<< endl;}
        kdtree.setInputCloud(ControlpointCloud);

        // ======================serialization============================
    }




    envLight::~envLight() {}

    pointEnvlight::pointEnvlight(const pointEnvlight & old) {
        pointEnvlight();
        envMapPose_world=old.envMapPose_world;
        pointBase= old.pointBase;
        EnvmapSampler.push_back(old.EnvmapSampler[0]);
        EnvmapSampler.push_back(old.EnvmapSampler[1]);
        ctrlPointIdx= old.ctrlPointIdx ;
    }


}

//  =================================== code notes=============================================


//        cout<<"show !!!!!!!!!"<< test_vec[0].pointBase << endl;
//        Vec3f key_test;
//        for ( auto key: envLightMap) {
//            cout<<"show keys:"<<key.first<<endl;
//            key_test=key.first;
//        }

//            cout<<"show !!!!!envLightMap!!!!"<< envLightMap.size() << endl;


//        test_vec.push_back(pEnv);


//            diffuseSampler = &pEnv.EnvmapSampler[1];
//            gli::vec4 SampleDiffuse = pEnv.EnvmapSampler[1].texture_lod(gli::fsampler2D::normalized_type(0.5, 1-0.75), 0.0f);
//                      cout << "\n============SampleDiffuse val(RGBA):\n" << SampleDiffuse.r << "," << SampleDiffuse.g << "," << SampleDiffuse.b << "," << SampleDiffuse.a << endl;
//        envLightMap.emplace(pEnv.pointBase, pEnv);
//            gli::vec4 SampleAAAAAA =pEnv.EnvmapSampler[0].texture_lod(gli::fsampler2D::normalized_type(0.5f,0.75f), 0.0f); // transform the texture coordinate
//            cout << "\n====IN1========SampleAAAAAA val(RGBA):\n"<< SampleAAAAAA.b << "," << SampleAAAAAA.g << "," <<SampleAAAAAA.r << "," << SampleAAAAAA.a << endl;
//
// check parameters.csv
//        char buff[256];
//        while (!parameters_file.eof()) {
//            parameters_file.getline(buff, 100);
//            string new_string =buff;
//            cout << new_string << endl;
//        }

//        cout<<"show  pEnv.pEnv.pointBase:"<< pEnv.pointBase<<endl;
//        cout<<"show pcl::PointXYZ(pEnv.pointBase.x, pEnv.pointBase.y, pEnv.pointBase.z)"<<pcl::PointXYZ(pEnv.pointBase.x, pEnv.pointBase.y, pEnv.pointBase.z)<<endl;


//        cout<<"show size of [firstPoint].EnvmapSampler:"<<envLightMap[firstPoint].EnvmapSampler.size()<<endl;
//        cout<<"show pointBase of [firstPoint].:"<<envLightMap[firstPoint].pointBase<<endl;
//            cout<<"\n Control pointCloud size:"<< ControlpointCloud->size()<<endl;
//            cout<<"\n show envLightMap size"<< envLightMap.size()<<endl;

//        cout<<"shwo  pEnv.envMapPose_world:"<< pEnv.envMapPose_world.matrix()<<endl;         //        pointEnv_Vec.push_back(pEnv);


//        brdfSampler_ = & brdfSampler[0];

//        vector<pointEnvlight> test_vec;
