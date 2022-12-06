//
// Created by cheng on 05.12.22.
//

#include "deltaCompute.h"

namespace DSONL
{

    float dot(const Eigen::Vector3f x, const Eigen::Vector3f y)
    {
        float ans = x.dot(y);
        return ans;
    }

    float _dot(Vec3f&  fst, Vec3f&  snd){ return max( (float)0.0, fst.dot(snd));}

    float mod(const float numer, const float denom)
    {
        return std::fmod(numer, denom);
    }

    float clamp(const float x, const float min_v, const float max_v)
    {
        return std::min(std::max(x, min_v), max_v);
    }

    float pow(const float x, const float y)
    {
        return std::pow(x, y);
    }

    Eigen::Vector3f pow(const float x, const Eigen::Vector3f y_vec)
    {
        Eigen::Vector3f out_vec;
        for (int i = 0; i < 3; i++)
        {
            out_vec[i] = pow(x, y_vec[i]);
        }
        return out_vec;
    }

    Eigen::Vector3f pow(const Eigen::Vector3f x, const float y)
    {
        return pow(y, x);
    }

    Eigen::Vector3f pow(const Eigen::Vector3f x_vec, const Eigen::Vector3f y_vec)
    {
        Eigen::Vector3f out_vec;
        for (int i = 0; i < 3; i++)
        {
            out_vec[i] = pow(x_vec[i], y_vec[i]);
        }
        return out_vec;
    }

    Eigen::Vector3f normalize(const Eigen::Vector3f vec)
    {
        Eigen::Vector3f out_vec(vec);
        float sum_quare(0);
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

    float mix(const float x, const float y, const float a)
    {
        if (a > 1 || a < 0)
            throw std::invalid_argument("received value a not in interval(0,1)");
        return x * (1 - a) + y * a;
    }
    Eigen::Vector3f mix(const Eigen::Vector3f x_vec, const Eigen::Vector3f y_vec, const float a)
    {
        Eigen::Vector3f out_vec;
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
    Vec3f reflect(Vec3f I, Vec3f N)
    {
      Vec3f N_norm = normalize(N);
      Vec3f out_vec = I - 2 * _dot(N_norm, I) * N_norm;
      return out_vec;
    }


    IBL_Radiance::IBL_Radiance() {}
    IBL_Radiance::~IBL_Radiance() {}


    Vec2f IBL_Radiance::directionToSphericalEnvmap(Vec3f dir) {
      float s = 1.0 - mod(1.0 / (2.0*M_PI) * atan2(dir.val[1], dir.val[0]), 1.0);
      float t = 1.0 / (M_PI) * acos(-dir.val[2]);
      return Vec2f(s, t);
    }


    Vec3f IBL_Radiance::specularIBL(Vec3f F0, float roughness, Vec3f N,
                                    Vec3f V) {

      float NoV = clamp(_dot(N, V), 0.0, 1.0);
      Vec3f R = reflect(-V, N);
      Vec2f uv = directionToSphericalEnvmap(R);

//      gli::vec4 Sample_val = prefilteredEnvmapSampler->texture_lod(gli::fsampler2D::normalized_type(0.5f, 0.75f),0.0f); // transform the texture coordinate
//      cout << "\n============Sample_val val(RGBA):\n" << Sample_val.b << "," << Sample_val.g << "," << Sample_val.r << ","   << Sample_val.a << endl;
      //       return Vec3f(Sample_val.r, Sample_val.g, Sample_val.b);
      //       EnvMapLookup.prefilteredColor(0.5f,0.75f, 0.0f);
      //       Vec3f color=EnvMapLookup.prefilteredColor(0.5f,0.75f, 0.0f);
      //       cout<<"test color:"<<color<<endl;
      //       Vec3f prefilteredColor = textureLod(prefilteredEnvmap, uv, roughness*float(mipCount)).rgb;


      return cv::Vec3f();
    }
    Vec3f IBL_Radiance::diffuseIBL(Vec3f normal) { return cv::Vec3f(); }
    Vec3f IBL_Radiance::fresnelSchlick(float cosTheta, Vec3f F0) {
      return cv::Vec3f();
    }

    }
