//
// Created by lei on 06.05.23.
//

#ifndef NLDSO_PHOTOMETRICLOSS_DELTACOMPUTION_H
#define NLDSO_PHOTOMETRICLOSS_DELTACOMPUTION_H


#include "settings/preComputeSetting.h"
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <cmath>
#include <math.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <unordered_map>


#include "envLightPreProcessing.h"



using namespace cv;
namespace PBANL
{

    template<typename T>
    bool checkImageBoundaries(const Eigen::Matrix<T, 2, 1> &pixel, int width, int height) ;

    float dot(const Eigen::Vector3f, const Eigen::Vector3f);

    float mod(const float, const float);

    float clamp(const float, const float, const float);

    float pow(const float, const float);
    Eigen::Vector3f pow(const float, const Eigen::Vector3f);
    //        template typename  TEigen::Matrix<T, 3,1>;
    //        Vec3f
    Eigen::Vector3f pow(const Eigen::Vector3f, const float);
    Eigen::Vector3f pow(const Eigen::Vector3f, const Eigen::Vector3f);

    Eigen::Vector3f normalize(const Eigen::Vector3f);

    float mix(const float, const float, const float);
    Eigen::Vector3f mix(const Eigen::Vector3f, const Eigen::Vector3f, const float);

    Vec3f reflect(Vec3f, Vec3f);



    class IBL_Radiance
    {

    public:
        IBL_Radiance();
        ~IBL_Radiance();
        int mipCount=5;
        Vec3f prefilteredColor(float u, float v, float level
//                                       ,pointEnvlight pointEnvlight_cur
        );
        Vec2f brdfIntegration(float NoV,float roughness );
        Vec2f directionToSphericalEnvmap(Vec3f dir);
        Vec3f specularIBL(Vec3f F0, float roughness, Vec3f N, Vec3f V, const Eigen::Matrix3d Camera1_c2w,
                          Sophus::SE3f enterEnv_Rotation
//                                  ,
//                                  pointEnvlight pointEnvlight_cur
        );
        Vec3f specularIBLCheck(Vec3f F0, float roughness, Vec3f N, Vec3f V, const Eigen::Matrix3d Camera1_c2w);

        Vec3f RRTAndODTFit( Vec3f v);
        Vec3f ACESFilm(Vec3f radiance);

        Vec3f diffusity;
        Vec3f Specularity;

        Vec3f diffuseIBL(Vec3f normal);
        Vec3f fresnelSchlick(float cosTheta, Vec3f F0);
        Vec3f ibl_radiance_val;
        Vec3f solveForRadiance(Vec3f viewDir, Vec3f normal,
                               const float& roughnessValue,
                               const float& metallicValue,
                               const float &reflectance,
                               const Vec3f& baseColorValue,
                               const Eigen::Matrix3d Transformation_wc,
                               Sophus::SE3f enterEnv_Rotation
//                                       ,
//                                       pointEnvlight pointEnvlight_cur
        );

        //EnvLight->envLightMap[key4Search].EnvmapSampler[0]

    private:
    };

    void updateDelta(
            Sophus::SE3d& Camera1_c2w,
            envLightLookup* EnvLightLookup,
            float *statusMap,
            Sophus::SO3d& Rotation,
            Eigen::Matrix<double, 3, 1>& Translation,
            const Eigen::Matrix3f& K,
//            const Mat& image_baseColor,
            const Mat depth_map,
//            const Mat& image_metallic,
            const Mat& image_roughnes,
            Mat& deltaMap,
            Mat& newNormalMap,
            Mat pointOfInterest,
            string  renderedEnvMapPath,
            Mat envMapWorkMask,
            Mat& specularityMap_1,
            Mat& specularityMap_2
    );







}



#endif //NLDSO_PHOTOMETRICLOSS_DELTACOMPUTION_H
