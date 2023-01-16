//
// Created by lei on 09.01.23.
//


#include "envLightPreProcessing.h"
#include <tbb/parallel_for.h>
#include <vector>
#include <mutex>

#include <cmath>

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

        for (size_t i=1;i<=fileNames.size();i++) {

            // only use selected area

            if (selectedIndex.count((int)i)==0){continue;}

//            if ((int)i!=seletcEnvMap || (int)i!=seletcEnvMap2  ){
//                continue;
//            }

            cout<<"\n =======================================show envMap index:"<< i<<endl;

            stringstream ss;
            string img_idx_str;
            ss << i;
            ss >> img_idx_str;
            string name_prefix = "/envMap";

            string  envMap_parameter_path = envMap_Folder+ name_prefix+img_idx_str+ "/parameters.csv";
            string  envMap_diffuse_parameter_path = envMap_Folder+ name_prefix+img_idx_str+ "/parameters_env_diffuse.csv";

            fstream parameters_file(envMap_diffuse_parameter_path);
            if (!parameters_file.is_open()){cout << "Error open shader_txt" << endl;}


            pointEnvlight pEnv;

            pEnv.envMapPose_world= controlPointPoses[i-1].cast<float>();
            pEnv.pointBase= Vec3f (  controlPointPoses[i-1].translation().x(), controlPointPoses[i-1].translation().y(),  controlPointPoses[i-1].translation().z());

            cout<<"\n =======================================show  pEnv.pointBase:"<<   pEnv.pointBase<<endl;

//            {
//                std::lock_guard<std::mutex>grd(mtx);

                //              envMap_parameter_path = "include/EnvLight_Data/envMap01/parameters.csv";// !!!!!!!temp!!!!!!!!!!1
            EnvMapLookup *EnvMapLookup=new DSONL::EnvMapLookup(argc,argv, envMap_parameter_path);
            EnvMapLookup->makeMipMap( pEnv.EnvmapSampler); // index_0: prefiltered Env light
            delete EnvMapLookup;
            //            envMap_diffuse_parameter_path = "include/EnvLight_Data/envMap01/parameters_env_diffuse.csv";// !!!!!!!temp!!!!!!!!!!1

            diffuseMap *diffuseMap = new DSONL::diffuseMap;
            diffuseMap->Init(argc,argv, envMap_diffuse_parameter_path);
            diffuseMap->makeDiffuseMap(pEnv.EnvmapSampler); // index_1: diffuse
            delete diffuseMap;

            ControlpointCloud->push_back(pcl::PointXYZ(pEnv.pointBase.x, pEnv.pointBase.y, pEnv.pointBase.z));
            envLightMap.insert(make_pair(pEnv.pointBase, pEnv));
            counter+=1;
            cout<<"\n show current envMap index: "<< i<<endl;
            cout<<"show number of envMap added: "<< counter<<endl;

//            }


    }



        if (ControlpointCloud->empty()){std::cerr<<"\n Wrong Control-pointCloud!"<< endl;}
        kdtree.setInputCloud(ControlpointCloud);

        // ======================serialization============================






    }




    envLight::~envLight() {

    }

    pointEnvlight::pointEnvlight(const pointEnvlight & old) {
        pointEnvlight();
        envMapPose_world=old.envMapPose_world;
        pointBase= old.pointBase;
        EnvmapSampler.push_back(old.EnvmapSampler[0]);
        EnvmapSampler.push_back(old.EnvmapSampler[1]);
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
