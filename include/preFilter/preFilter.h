//
// Created by cheng on 19.11.22.
//


#ifndef NLDSO_PHOTOMETRICLOSS_PREFILTER_H
#define NLDSO_PHOTOMETRICLOSS_PREFILTER_H



//#pragma once

#include "preFilterUtils.h"
#include "dataLoader/PFMReadWrite.h"


namespace DSONL {

	class preFilterSpecularMask {

	public:
		explicit preFilterSpecularMask(float &roughness);
		~preFilterSpecularMask();
		float _roughness;
		Mat outColor;
		Mat envMapImage;
		void Init(int argc, char **argv, string parameter_path);
	};


	class preFilter {
	public:
		preFilter(float &roughness);
		~preFilter();
		float _roughness;
		Mat outColor;
		Mat envMapImage;
		// void modify_shader_file(string, string) const;
		void Init(int argc, char **argv,string parameter_path);// int argc, char** argv
	private:
		Eigen::Matrix<float, 2, 1> tc;
		int samples;
		float envMapLevel;
	};

	//        void preFilter::modify_shader_file(string roughness_str, string
	//        envmapLvl_str) const
	//        {
	//          string shader_txt = "../include/shaders/fragment_shader.txt";
	//          string shader_txt_out =
	//          "../include/shaders/fragment_shader_modified.txt"; char buff[256];
	//          std::cmatch m;
	//          fstream setRoughness(shader_txt);
	//          ofstream setRoughness_out(shader_txt_out, ios::out);
	//          regex rule_roughness("(roughness.*defaultval)");
	//          regex rule_envmapLvl("(envMapLevel.*defaultval)");
	//          regex rule_default("defaultval(\\s*)=(\\s*)\".*\"");
	//          if (!setRoughness.is_open())
	//            cout << "Error open shader_txt" << endl;
	//          if (!setRoughness_out.is_open())
	//            cout << "Error open shader_txt_out" << endl;
	//          while (!setRoughness.eof())
	//          {
	//            setRoughness.getline(buff, 100);
	//            bool ret_1 = std::regex_search(buff, m, rule_roughness);
	//            bool ret_2 = std::regex_search(buff, m, rule_envmapLvl);
	//            if (ret_1)
	//            {
	//              // cout << buff << endl;
	//              char toreplace_buff[256];
	//              sprintf(toreplace_buff, "defaultval=\"%s\"",
	//              roughness_str.c_str()); string new_string =
	//              std::regex_replace(buff, rule_default, toreplace_buff);
	//              setRoughness_out << new_string << endl;
	//            }
	//
	//            else if (ret_2)
	//            {
	//              // cout << buff << endl;
	//              char toreplace_buff[256];
	//              sprintf(toreplace_buff, "defaultval=\"%s\"",
	//              envmapLvl_str.c_str()); string new_string =
	//              std::regex_replace(buff, rule_default, toreplace_buff);
	//              setRoughness_out << new_string << endl;
	//            }
	//            else
	//            {
	//              setRoughness_out << buff << endl;
	//            }
	//          }
	//          setRoughness.close();
	//          setRoughness_out.close();
	//        }
	//


	class EnvMapLookup {
	public:
        EnvMapLookup();
		EnvMapLookup(int argc, char **argv, string parameter_path);
		~EnvMapLookup();
		std::vector<cv::Mat> image_pyramid_mask;
		std::vector<cv::Mat> image_pyramid;
		cv::Mat applyMask(cv::Mat &input, cv::Mat &mask);
		void mergeImagePyramid();
		void makeMipMap(std::vector<gli::sampler2d<float>>& Sampler_vec, string env_path);

	private:
	};


    class brdfIntegrationMap {

    public:
        brdfIntegrationMap(string map_path);
        ~brdfIntegrationMap();
        void makebrdfIntegrationMap( std::vector<gli::sampler2d<float>> &brdf);
        Mat brdfIntegration_Map;

        string brdf_path;

    private:
    };

	class diffuseMapMask {
	public:
		diffuseMapMask();
		~diffuseMapMask();
		void Init(int argc, char **argv,string parameters_path);
		cv::Mat diffuse_Map_Mask;
		void makeDiffuseMask();

	private:
	};


	class diffuseMap {

	public:
		diffuseMap();
		~diffuseMap();

		void Init(int argc, char **argv, string parameters_path);
		cv::Mat diffuse_Map;
		cv::Mat diffuse_MaskMap;
		cv::Mat diffuse_Final_Map;
		void mergeDiffuse_Map();
		void makeDiffuseMap(std::vector<gli::sampler2d<float>>& Sampler_diffuse_vec, string envDiffuse_path );

		cv::Mat applyMask(Mat &input, Mat &mask);
		gli::vec4 getDiffuse(float u, float v);

	private:

	};


}// namespace DSONL


#endif //NLDSO_PHOTOMETRICLOSS_PREFILTER_H




// ======================convert the BGR to RGB==============================
//	cv::cvtColor(image_pyramid[0], image_pyramid[0], COLOR_BGR2RGB);

//  experiment notes:

//		gli::extent2d cood(512,384);
//		gli::vec4 Sampletest= Sampler_single.texel_fetch(cood,0);
//		cout<<"\n============Sampletest
//val(RGBA):\n"<<Sampletest.b<<","<<Sampletest.g<<","<<Sampletest.r<<","<<Sampletest.a<<
//endl;

// Good test case 1:
// GLI fetch textel directly:
//  fetch texel using (col, row) coordinate
//		gli::extent2d cood(512,256);
//		============Sampletest val(RGBA):
//		0.172549,0.0901961,0.0823529,1

// gsn composer  0.5,0.5,0:
//		image_ref_path_PFM blue  green and red channel value:
//		[0.0870098, 0.0894608, 0.168137]

// textureLod 0.5, 0.5, 0;
//		============SampleA val(RGBA):
//		0.167647,0.0892157,0.0862745,1

// Good test case 2:
// GLI fetch textel directly:
//  fetch texel using (col, row) coordinate
//		gli::extent2d cood(512,128);
//		============Sampletest val(RGBA):
//		0.0392157,0.0235294,0.00784314,1

// gsn composer  0.5,0.25,0:
// image_ref_path_PFM blue  green and red channel value:
// [0.0387254, 0.142647, 0.312745]

// textureLod 0.5, 0.25, 0;
//		============SampleA val(RGBA):
//		0.0421569,0.0254902,0.00833333,1

// case 3

//		gli::extent2d cood(512,384);(col, row) coordinate
//		============Sampletest val(RGBA):
//		0.305882,0.141176,0.0392157,1

// gsn composer  0.5,0.25,0:
// image_ref_path_PFM blue  green and red channel value:
// [0.0387254, 0.142647, 0.312745]
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1!For
// comparision!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! textureLod 0.5, 0.75,
// 0;
//		============SampleA val(RGBA):
//		0.318627,0.145588,0.0387255,1
//============Sample_val val(BGRA):
//    0.0627451,0.0470588,0.0215686,1

// =========================save data=====================================

//		std::string outputFile = "outputFile.dds";
//		gli::save(newTex, outputFile);
//		glm::highp_float32 TexelA =
//gli::textureLod<glm::highp_float32>(newTex,
//gli::texture2D::texcoord_type(0.0f, 0.0f), 0);

// =========================================================note==========================================

//		// checking
//		cv::Mat checkMemcpy(512, 1024, CV_32FC3);
//		memcpy(checkMemcpy.data, orig_tex.data(), orig_tex.size());
//		//		cv::cvtColor(checkMemcpy, checkMemcpy,
//COLOR_RGB2BGR) ; 		imshow("checkMemcpy", checkMemcpy);
//
//		waitKey(0);
//		gli::texture2d  env_map_texture(tex.format(), tex.extent(),
//numMipmaps);
//
//
////		std::vector<gli::image2D> pyramid_Images;
//
//		for (int i = 0; i < 6; ++i) {
//
//			cv::cvtColor(image_pyramid[i], image_pyramid[i],
//COLOR_BGR2RGB) ; 			cv::Mat flat = image_pyramid[i].reshape(1,
//image_pyramid[i].total() * image_pyramid[i].channels()); 			std::vector<float>
//img_array = image_pyramid[i].isContinuous() ? flat : flat.clone(); 			glm::uvec2
//dim(image_pyramid[i].cols,image_pyramid[i].cols); // width, height
//
//			gli::texture2d env_map_texture();
////			gli::image2D  image_lvl_i(dim, gli::RGB32F, img_array);
////			pyramid_Images.push_back(image_lvl_i);
//
//		}
//
//
////		gli::texture2d tex(format, gli::texture2d::extent_type(Width,
///Height), 1); /		memcpy(tex.data(), FreeImage_GetBits(bmp),
///tex.size());
//
//
////		image_lvl_i
//
////		gli::texture2D Mipmap;
//
//

//		glGenTextures(1, &EnvMapTexture);
//		glBindTexture(GL_TEXTURE_2D, EnvMapTexture);

//		glTexImage2D(pyramid,)

//                float NoV= 0.5f, roughness= 0.25f;

//		gli::vec4 SampleBrdf =
//Sampler_brdf.texture_lod(gli::fsampler2D::normalized_type(0.5f,0.25f), 0.0f);
//// transform the texture coordinate 		cout << "\n============SampleBrdf
//val(RGBA):\n"
//			 << SampleBrdf.b << "," << SampleBrdf.g << "," <<
//SampleBrdf.r << "," << SampleBrdf.a << endl;
//
//
//                gli::vec4 SampleBrdf_test =
//                brdfSampler->texture_lod(gli::fsampler2D::normalized_type(0.5f,0.25f),
//                0.0f); // transform the texture coordinate cout <<
//                "\n============SampleBrdf_test val(RGBA):\n"
//                     << SampleBrdf_test.b << "," << SampleBrdf_test.g << ","
//                     << SampleBrdf_test.r << "," << SampleBrdf_test.a << endl;
//

// case 1
// 0.5, 0.5
//  image_ref_path_PFM blue,  green and red channel value: [0, 0.0179544,
//  0.724845]
// ============SampleBrdf val(RGBA):
// 0.7253,0.0186449,0,1
// case 2
// gsn : 0.5, 0.75
//		image_ref_path_PFM blue  green and red channel value: [0,
//0.00670501, 0.570426]
// GLI: 0.5, 0.25
//		============SampleBrdf val(BGRA):  0.573688,0.00701116,0,1
