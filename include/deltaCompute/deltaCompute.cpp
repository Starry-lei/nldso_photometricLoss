//
// Created by cheng on 05.12.22.
//

#include "deltaCompute.h"

namespace DSONL
{

    double dot(const Eigen::Vector3d x, const Eigen::Vector3d y)
    {
        double ans = x.dot(y);
        return ans;
    }

    double mod(const double numer, const double denom)
    {
        return fmod(numer, denom);
    }

    double clamp(const double x, const double min_v, const double max_v)
    {
        return std::min(std::max(x, min_v), max_v);
    }

    double pow(const double x, const double y)
    {
        return std::pow(x, y);
    }

    Eigen::Vector3d pow(const double x, const Eigen::Vector3d y_vec)
    {
        Eigen::Vector3d out_vec;
        for (int i = 0; i < 3; i++)
        {
            out_vec[i] = pow(x, y_vec[i]);
        }
        return out_vec;
    }

    Eigen::Vector3d pow(const Eigen::Vector3d x, const double y)
    {
        return pow(y, x);
    }

    Eigen::Vector3d pow(const Eigen::Vector3d x_vec, const Eigen::Vector3d y_vec)
    {
        Eigen::Vector3d out_vec;
        for (int i = 0; i < 3; i++)
        {
            out_vec[i] = pow(x_vec[i], y_vec[i]);
        }
        return out_vec;
    }

    Eigen::Vector3d normalize(const Eigen::Vector3d vec)
    {
        Eigen::Vector3d out_vec(vec);
        double sum_quare(0);
        for (int i = 0; i < 3; i++)
        {
            sum_quare += vec[i] * vec[i];
        }
        if (sum_quare < 1e-10f)
        {
            printf("[WARNING] The norm of vector is too small, return zeros vector.\n");
            return out_vec; // too small norm
        }
        sum_quare = std::sqrt(sum_quare);
        for (int i = 0; i < 3; i++)
        {
            out_vec[i] /= sum_quare;
        }
        return out_vec;
    }

    double mix(const double x, const double y, const double a)
    {
        if (a > 1 || a < 0)
            throw std::invalid_argument("received value a not in interval(0,1)");
        return x * (1 - a) + y * a;
    }
    Eigen::Vector3d mix(const Eigen::Vector3d x_vec, const Eigen::Vector3d y_vec, const double a)
    {
        Eigen::Vector3d out_vec;
        for (int i = 0; i < 3; i++)
        {
            out_vec[i] = mix(x_vec[i], y_vec[i], a);
        }
        return out_vec;
    }

    /*
        Reflect — calculate the reflection direction for an incident vector
        I: Specifies the incident vector.
        N: Specifies the normal vector.
    */
    Eigen::Vector3d reflect(const Eigen::Vector3d I, const Eigen::Vector3d N)
    {
        Eigen::Vector3d N_norm = normalize(N);
        Eigen::Vector3d out_vec = I - 2 * dot(N_norm, I) * N_norm;
        return out_vec;
    }
    IBL_Radiance::IBL_Radiance() {}
    IBL_Radiance::~IBL_Radiance() {}
    Vec2f IBL_Radiance::directionToSphericalEnvmap(Vec3f dir) {
      return cv::Vec2f();
    }
    Vec3f IBL_Radiance::specularIBL(Vec3f F0, float roughness, Vec3f N,
                                    Vec3f V) {
      return cv::Vec3f();
    }
    Vec3f IBL_Radiance::diffuseIBL(Vec3f normal) { return cv::Vec3f(); }
    Vec3f IBL_Radiance::fresnelSchlick(float cosTheta, Vec3f F0) {
      return cv::Vec3f();
    }

    }
