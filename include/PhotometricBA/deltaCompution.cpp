//
// Created by lei on 06.05.23.
//

#include "deltaCompution.h"

namespace PBANL{


    float dot(const Eigen::Vector3f x, const Eigen::Vector3f y) {
        float ans = x.dot(y);
        return ans;
    }

    float clamp(const float x, const float min_v, const float max_v) {
        return std::min(std::max(x, min_v), max_v);
    }

    Vec3f  clamp_vec3f(Vec3f x){

        Vec3f res;
        res.val[0]= clamp(x.val[0], 0.0f, 1.0f);
        res.val[1]= clamp(x.val[1], 0.0f, 1.0f);
        res.val[2]= clamp(x.val[2], 0.0f, 1.0f);

        return res;
    }


    float _dot(Vec3f &fst, Vec3f &snd) { return max((float) 0.0, fst.dot(snd)); }
    float mod(const float numer, const float denom) {
        return std::fmod(numer, denom);
    }
    float glslmod(const float numer, const float denom) {
        return numer - denom * std::floor(numer / denom);
    }


    Vec3f pow(const float x, Vec3f y_vec) {
        Vec3f out_vec;
        for (int i = 0; i < 3; i++) {
            out_vec[i] = std::pow(x, y_vec[i]);
        }
        return out_vec;
    }
    Vec3f pow(const Vec3f x_vec, const Vec3f y_vec) {
        Vec3f out_vec;
        for (int i = 0; i < 3; i++) {
            out_vec[i] = std::pow(x_vec[i], y_vec[i]);
        }
        return out_vec;
    }
    float mix(const float x, const float y, const float a) {
        if (a > 1 || a < 0)
            throw std::invalid_argument("received value a not in interval(0,1)");
        return x * (1 - a) + y * a;
    }
    Vec3f mix(const Vec3f x_vec, const Vec3f y_vec, const float a) {
        Vec3f out_vec;
        for (int i = 0; i < 3; i++) {
            out_vec[i] = mix(x_vec[i], y_vec[i], a);
        }
        return out_vec;
    }

    /*
        Reflect — calculate the reflection direction for an incident vector
        I: Specifies the incident vector.
        N: Specifies the normal vector.
    */
    Vec3f reflect(Vec3f I, Vec3f N) {


        Vec3f N_norm = cv::normalize(N);
        Vec3f out_vec = I - 2.0f * N_norm.dot(I) * N_norm;

        return out_vec;
    }

    IBL_Radiance::IBL_Radiance() {}
    IBL_Radiance::~IBL_Radiance() {}

    Vec2f IBL_Radiance::directionToSphericalEnvmap(Vec3f dir) {

//		float s = 1.0 - glslmod(1.0 / (2.0 * M_PI) * atan2(dir.val[1], dir.val[0]), 1.0);
        float s =  glslmod(1.0 / (2.0 * M_PI) * atan2(-dir.val[1], dir.val[0]), 1.0);
        //      float s = 1.0 - mod(1.0 / (2.0*M_PI) * atan2(dir.val[1], dir.val[0]), 1.0);
        float t = 1.0 / (M_PI) *acos(-dir.val[2]);
        if (s > 1.0 || t > 1.0) { std::cerr << "UV coordinates overflow!" << std::endl; }

        return Vec2f(s, t);
    }

    Vec3f IBL_Radiance::specularIBL(Vec3f F0, float roughness, Vec3f N, Vec3f V, const Eigen::Matrix3d Camera1_c2w,
                                    Sophus::SO3f enterEnv_Rotatio_inv
    ) {

        float NoV = clamp(_dot(N, V), 0.0, 1.0);

        // Vec3f R =reflect(-V, N);
        // std::cout<<"show the norm of R"<<norm(R)<<std::endl;
        // Vec2f uv = directionToSphericalEnvmap(R);

        Vec3f R_c = reflect(-V, N);//
        // -0.927108467, 0.132981807, -0.350408018
        Eigen::Matrix<float, 3, 1> R_c_(R_c.val[0], R_c.val[1], R_c.val[2]);
        // convert coordinate system
        Eigen::Vector3f R_w =  enterEnv_Rotatio_inv.matrix()*Camera1_c2w.cast<float>() * R_c_;


        Vec2f uv = directionToSphericalEnvmap(Vec3f(R_w.x(), R_w.y(), R_w.z()));
        if (uv.val[0] > 1.0 || uv.val[1] > 1.0) {
            std::cerr << "\n===specularIBL=======Show UV=================:" << uv << std::endl;
        }


        //      std::cout<<"uv:"<<uv<<"show roughness*float(mipCount)"<<roughness*float(mipCount)<<std::endl;
        Vec3f prefiltered_Color = prefilteredColor(uv.val[0], uv.val[1], roughness * float(mipCount)
//                                                   ,pointEnvlight_cur
        );

        //    uv:[0.179422, 0.400616]show roughness*float(mipCount)1.5
        //          prefiltered_Color:[0.112979, 0.0884759, 0.0929931]

        //      show image_ref_path_PFM  of GSN(293,476)And values: [5.17454, 4.71557, 0.0619548]


        Vec3f prefiltered_Color_check = prefiltered_Color;

        Vec2f brdf_val = brdfIntegration(NoV, roughness);
        // TODO: use matrix operator here
        F0.val[0] *= brdf_val.val[0];
        F0.val[1] *= brdf_val.val[0];
        F0.val[2] *= brdf_val.val[0];

        F0.val[0] += brdf_val.val[1];
        F0.val[1] += brdf_val.val[1];
        F0.val[2] += brdf_val.val[1];

//        cout<<"show T BRDF>>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<F0<<endl;
//        cout<<"\n show T u  and 1-v >>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<uv.val[0]<<","<< uv.val[1]<<":"<<" 1- uv.val[1]"<<endl;

        prefiltered_Color.val[0] *= F0.val[0];
        prefiltered_Color.val[1] *= F0.val[1];
        prefiltered_Color.val[2] *= F0.val[2];

//        cout<<"show T prefiltered_Color>>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<prefiltered_Color<<endl;
        return prefiltered_Color;
    }




    Vec3f IBL_Radiance::diffuseIBL(Vec3f normal) {

        Vec2f uv = directionToSphericalEnvmap(normal);

        if (uv.val[0] > 1.0 || uv.val[1] > 1.0) {
            std::cerr << "\n===diffuseIBL=======Show UV=================:" << uv << std::endl;
        }


        gli::vec4 Sample_val = DSONL::diffuseSampler->texture_lod(gli::fsampler2D::normalized_type(uv.val[0], 1 - uv.val[1]), 0.0f);// transform the texture coordinate

        // new formula                                                                                                  //      std::cout << "\n============diffuseIBL val(BGRA):\n"<<"diffuseIBL:"<< uv<<"diffuse val:"<< Sample_val.b << "," << Sample_val.g << "," << Sample_val.r << ","   << Sample_val.a << std:: endl;
        cv::Vec3f radiance_diff= cv::Vec3f((float) Sample_val.b, (float) Sample_val.g, (float) Sample_val.r);

        return radiance_diff;
    }

    Vec3f IBL_Radiance::fresnelSchlick(float cosTheta, Vec3f F0) {

        return (F0 + (Vec3f(1.0f, 1.0f, 1.0f) - F0) * std::pow(1.0f - cosTheta, 5.0f));
    }

    Vec3f IBL_Radiance::prefilteredColor(float u, float v, float level
//                                         ,pointEnvlight pointEnvlight_cur
    ) {
        gli::vec4 Sample_val = DSONL:: prefilteredEnvmapSampler->texture_lod(gli::fsampler2D::normalized_type(u, 1 - v), level);// transform the texture coordinate
        //      std::cout << "\n============Sample_val val(BGRA):\n" << Sample_val.b << "," << Sample_val.g << "," << Sample_val.r << ","   << Sample_val.a << std:: endl;
        return cv::Vec3f((float) Sample_val.b, (float) Sample_val.g, (float) Sample_val.r);
    }

    Vec2f IBL_Radiance::brdfIntegration(float NoV, float roughness) {
        gli::vec4 SampleBrdf = DSONL::brdfSampler_->texture_lod(gli::fsampler2D::normalized_type(NoV, 1 - roughness), 0.0f);// transform the texture coordinate
        //     std::cout << "\n============SampleBrdf val(BGRA)!!!!!!!:\n" << SampleBrdf.b << "," << SampleBrdf.g << "," << SampleBrdf.r << "," << SampleBrdf.a << std::endl;
        return Vec2f(SampleBrdf.b, SampleBrdf.g);                                                                   // x, y  SampleBrdf.b, SampleBrdf.g
    }

    Vec3f IBL_Radiance::solveForRadiance(Vec3f viewDir, Vec3f normal,
                                         const float &roughnessValue,
                                         const float &metallicValue,
                                         const float &reflectance,
                                         const Vec3f &baseColorValue,
                                         const Eigen::Matrix3d Camera1_c2w,
                                         Sophus::SO3f enterEnv_Rotation_inv) {

        // !!!!!!!!  vec3 baseCol = pow(texture(baseColorTexture, texScale*tc).rgb, vec3(2.2)); // this is gamma correction!
        Vec3f One = Vec3f(1.0f, 1.0f, 1.0f);
        Vec3f f0 = 0.16 * (reflectance * reflectance) * One;
        f0 = mix(f0, baseColorValue, metallicValue);
        Vec3f F = fresnelSchlick(_dot(normal, viewDir), f0);
        Vec3f kS = F;
        Vec3f kD = One - kS;
        kD = kD.mul(One - metallicValue * One);
        Vec3f specular = specularIBL(f0, roughnessValue, normal, viewDir,Camera1_c2w,enterEnv_Rotation_inv);
//        Vec3f specular = specularIBL(F, roughnessValue, normal, viewDir,Camera1_c2w,enterEnv_Rotation_inv);// !!!!!
        Specularity = specular;
        //convert from camera to world
        Eigen::Vector3d normal_c(normal.val[0], normal.val[1], normal.val[2]);
        Vec3f normal_w((Camera1_c2w * normal_c).x(), (Camera1_c2w * normal_c).y(), (Camera1_c2w * normal_c).z());
//        Vec3f diffuse = diffuseIBL(normal_w);
//        diffusity=diffuse;
        // only focus on specular property
        Vec3f diffuse=Vec3f(0.0,0.0,0.0);
//        cout<<"Checking kD:"<<kD<<endl;
        // shading front-facing
        Vec3f color = pow(kD.mul(baseColorValue.mul(diffuse)) + specular, 1.0 / 2.2 * One);

//        cout<<"\n Checking vals:"<<"kD: "<<kD <<","<<"baseColorValue: "<< baseColorValue<< ","<<"diffuse: "<<diffuse<<","<<"specular: "<<specular<<endl;
//        cout<<"Checking color:"<<color<<endl;

        //      Vec3f color=diffuse;


//		// shading back-facing
//		if (viewDir.dot(normal) < -0.1) {
//			//discard;
//			color = 0.1 * baseColorValue.mul(diffuse);
//			//        std::cerr<<"_dot(viewDir, normal) < -0.1:"<< color<<std::endl;
//		}

        return color;
    }

    Vec3f IBL_Radiance::RRTAndODTFit(Vec3f v) {
        Vec3f One = Vec3f(1.0f, 1.0f, 1.0f);

        Vec3f a = v .mul(v + 0.0245786f*One ) - 0.000090537f*One;
        Vec3f b = v.mul (0.983729f * v + 0.4329510f*One) + 0.238081f*One;

        Vec3f inv_v= Vec3f (1.0f/b.val[0],  1.0f/b.val[1],1.0f/b.val[2] );

        return a.mul(inv_v);

    }

    Vec3f IBL_Radiance::ACESFilm(Vec3f radiance) {

        // https://computergraphics.stackexchange.com/questions/11018/how-to-change-a-rgb-value-using-a-radiance-value
        // https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
        // method 1

        float adapted_lum=1.0;
        radiance*= adapted_lum;
        Eigen::Matrix3f ACESInputMat, ACESOutputMat;
        Eigen::Vector3f r1, r2;
        ACESInputMat <<  0.59719, 0.35458, 0.04823, 0.07600, 0.90834, 0.01566, 0.02840, 0.13383, 0.83777;
        ACESOutputMat <<  1.60475, -0.53108, -0.07367 , -0.10208,  1.10813, -0.00605 ,-0.00327, -0.07276,  1.07602;

        r1= ACESInputMat* Eigen::Vector3f(radiance.val[0], radiance.val[1],radiance.val[2] );
        radiance=RRTAndODTFit(Vec3f(r1.x(),r1.y(),r1.z()));
        r2= ACESOutputMat*Eigen::Vector3f(radiance.val[0], radiance.val[1],radiance.val[2] );

        radiance=Vec3f(r2.x(),r2.y(),r2.z());
        // method 2
        //			float a = 2.51f;
        //			float b = 0.03f;
        //			float c = 2.43f;
        //			float d = 0.59f;
        //			float e = 0.14f;
        //		    Vec3f One = Vec3f(1.0f, 1.0f, 1.0f);
        //		    Vec3f sndVecor= Vec3f(1.0/(radiance.mul(c*radiance+d*One)+e*One).val[0],
        //		                           1.0/(radiance.mul(c*radiance+d*One)+e*One).val[1],
        //		                           1.0/(radiance.mul(c*radiance+d*One)+e*One).val[2]);
        //		return  clamp_vec3f((radiance.mul(a*radiance+b*One)) .mul(sndVecor));

        // method 1 output
        return clamp_vec3f(radiance);

    }


    float calculateCosDis(Vec3f vec1, Vec3f vec2){
        return vec1.dot(vec2);
    }

    template<typename T>
    bool checkImageBoundaries(const Eigen::Matrix<T, 2, 1> &pixel, int width, int height) {
        return (pixel[0] > 1.1 && pixel[0] < width - 2.1 && pixel[1] > 1.1 && pixel[1] < height - 2.1);
    }




}