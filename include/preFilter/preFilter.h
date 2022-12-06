//
// Created by cheng on 19.11.22.
//

#pragma once

#include <GL/glew.h>
#include <GL/freeglut.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <Eigen/Core>
#include <memory>

#include "preComputeSetting.h"
#include "dataLoader/PFMReadWrite.h"


// GLI library
#define GLM_ENABLE_EXPERIMENTAL
#include "gli/gli/gli.hpp"
#include "gli/gli/texture2d.hpp"
#include "gli/gli/sampler2d.hpp"

#include "preFilter/Renderer.h"
#include "diffuseMap/Renderer_diffuse.h"
#include "diffuseMap/diffuse_mask_Renderer.h"
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace gsn;
using namespace cv;

namespace DSONL
{

	static Renderer *renderer = NULL; //this is a static pointer to a Renderer used in the glut callback functions
	static Renderer_diffuse *render_diffuse = NULL;
	static diffuse_Mask_Renderer *render_diffuse_mask = NULL;




	int window(0);
	int window_diffuse(0);
	int window_diffuse_mask(0);

	//glut static callbacks start
	static void glutResize(int w, int h)
	{
		if (renderer->roughness == (float)0.0)
		{
			glutReshapeWindow(1024, 512);
		}
		else if (renderer->roughness == (float)0.2)
		{
			glutReshapeWindow(512, 256);
		}
		else if (renderer->roughness == (float)0.4)
		{
			glutReshapeWindow(256, 128);
		}
		else if (renderer->roughness == (float)0.6)
		{
			glutReshapeWindow(128, 64);
		}
		else if (renderer->roughness == (float)0.8)
		{
			glutReshapeWindow(64, 32);
		}
		else if (renderer->roughness == (float)1.0)
		{
			glutReshapeWindow(32, 16);
		}
		else
		{
			std::cerr << "Wrong roughness!" << endl;
		}
	}

	//glut static callbacks start
	static void glutResize_diffuse(int w, int h)
	{
		render_diffuse->resize(w, h);
	}

	static void glutResize_diffuse_mask(int w, int h)
	{
		render_diffuse_mask->resize(w, h);
	}

	static void glutDisplay()
	{
		renderer->display();
		glutSwapBuffers();
		glutReportErrors();
		glutDestroyWindow(window);
	}

	static void glutDisplay_diffuse()
	{
		render_diffuse->display();
		glutSwapBuffers();
		glutReportErrors();
		glutDestroyWindow(window_diffuse);
	}
	static void glutDisplay_diffuse_mask()
	{
		render_diffuse_mask->display();
		glutSwapBuffers();
		glutReportErrors();
		glutDestroyWindow(window_diffuse_mask);
	}

	//	render_diffuse

	static void glutClose()
	{
		renderer->dispose();
		delete renderer;
	}
	static void glutClose_diffuse()
	{
		render_diffuse->dispose();
		delete render_diffuse;
	}
	static void glutClose_diffuse_mask()
	{
		render_diffuse_mask->dispose();
		delete render_diffuse_mask;
	}

	static void timer(int v)
	{
		float offset = 1.0f;
		renderer->t += offset;
		glutDisplay();
		glutTimerFunc(unsigned(20), timer, ++v);
	}
	static void timer_diffuse(int v)
	{
		float offset = 1.0f;
		render_diffuse->t += offset;
		glutDisplay_diffuse();
		glutTimerFunc(unsigned(20), timer_diffuse, ++v);
	}
	static void timer_diffuse_mask(int v)
	{
		float offset = 1.0f;
		render_diffuse_mask->t += offset;
		glutDisplay_diffuse_mask();
		glutTimerFunc(unsigned(20), timer_diffuse_mask, ++v);
	}

	static void glutKeyboard(unsigned char key, int x, int y)
	{
		bool redraw = false;
		std::string modeStr;
		std::stringstream ss;
		if (key >= '1' && key <= '9')
		{
			renderer->selectedOutput = int(key) - int('1');
		}
	}

	static void glutKeyboard_diffuse(unsigned char key, int x, int y)
	{
		bool redraw = false;
		std::string modeStr;
		std::stringstream ss;
		if (key >= '1' && key <= '9')
		{
			render_diffuse->selectedOutput = int(key) - int('1');
		}
	}

	static void glutKeyboard_diffuse_mask(unsigned char key, int x, int y)
	{
		bool redraw = false;
		std::string modeStr;
		std::stringstream ss;
		if (key >= '1' && key <= '9')
		{
			render_diffuse_mask->selectedOutput = int(key) - int('1');
		}
	}

	class preFilter
	{
            public:
                    preFilter(float &roughness);
                    ~preFilter();
                    float _roughness; // description="roughness in range [0.0, 1.0]" defaultval ="0.2"
                    Mat outColor;
                    Mat envMapImage;
                    void Init(int argc, char **argv); //int argc, char** argv
            private:
                    Eigen::Matrix<float, 2, 1> tc; // texture coordinate of the output image in range [0.0, 1.0]
                    int samples;				   // description="number of samples" defaultval="64"
                    float envMapLevel;			   // description="Environment map level" defaultval="0.0"
	};

	preFilter::preFilter(float &roughness)
	{
		_roughness = roughness;
		string shader_txt = "../include/shaders/fragment_shader.txt";
		fstream setRoughness(shader_txt);
		std::ostringstream oss;
		oss << std::setprecision(1) << _roughness;
		string roughness_str;
		(_roughness == 1 || _roughness == 0) ? roughness_str = oss.str() + ".0" : roughness_str = oss.str();
//		cout << "show roughness_str:" << roughness_str << endl;
		// set corresponding env_map level
		float envmapLvl;
		roughness == (float)0.2 ? envmapLvl = 0.0 : envmapLvl = roughness * 5.0;
		std::ostringstream oss_envl;
		oss_envl << std::setprecision(2) << envmapLvl;
		string envmapLvl_str = oss_envl.str() + ".0";
//		cout << "envmapLvl_str:" << envmapLvl_str << endl;
		setRoughness.seekg(221L, ios::beg);
		setRoughness << roughness_str;
		setRoughness.seekg(378L, ios::beg);
		setRoughness << envmapLvl_str;
		setRoughness.close();
	}
	preFilter::~preFilter() {}

	void preFilter::Init(int argc, char **argv)
	{
		renderer = new Renderer;
		renderer->roughness = _roughness;
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
		glutInitWindowPosition(100, 100);
		glutInitWindowSize(1024, 512);
		window = glutCreateWindow("GSN Composer Shader Export");
		glutHideWindow();
		GLenum err = glewInit();
		if (GLEW_OK != err)
		{
			fprintf(stderr, "Glew error: %s\n", glewGetErrorString(err));
		}
		glutDisplayFunc(glutDisplay);
		glutReshapeFunc(glutResize);
		glutCloseFunc(glutClose);
		glutKeyboardFunc(glutKeyboard);

		if (_roughness == (float)0.0)
		{
			renderer->resize(1024, 512);
		}
		else if (_roughness == (float)0.2)
		{
			renderer->resize(512, 256);
		}
		else if (_roughness == (float)0.4)
		{
			renderer->resize(256, 128);
		}
		else if (_roughness == (float)0.6)
		{
			renderer->resize(128, 64);
		}
		else if (_roughness == (float)0.8)
		{
			renderer->resize(64, 32);
		}
		else if (_roughness == (float)1.0)
		{
			renderer->resize(32, 16);
		}
		else
		{
			std::cerr << "Wrong roughness!" << endl;
		}
		renderer->init();
		glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
		glutTimerFunc(unsigned(20), timer, 0);
		glutMainLoop();
	}

	class EnvMapLookup
	{
	public:
		EnvMapLookup(int argc, char **argv);
		~EnvMapLookup();
		std::vector<cv::Mat> image_pyramid;
		void prefilteredColor(float u, float v, float level);
                gli::sampler2d<float>* makeMipMap( );
	private:


	};

	EnvMapLookup::EnvMapLookup(int argc, char **argv)
	{

		float roughness_i = 0.0;
		for (int i = 0; i < 6; ++i)
		{
			roughness_i = i / 5.0;
			preFilter env_mpa_preFilter(roughness_i);
			env_mpa_preFilter.Init(argc, argv);
		}
		image_pyramid = img_pyramid;
	}

	EnvMapLookup::~EnvMapLookup()
	{
	}

        gli::sampler2d<float>*  EnvMapLookup::makeMipMap()
	{

		//===================================================mind  opencv BGR order  ...=================================
		// define and allocate space for Mipmap using GLM library
		std::size_t numMipmaps = 6;
		gli::texture2d orig_tex(gli::FORMAT_RGB32_SFLOAT_PACK32, gli::texture2d::extent_type(1024, 512), 1);
		cv::Mat flat = image_pyramid[0].reshape(1, image_pyramid[0].total() * image_pyramid[0].channels());
		std::vector<float> img_array = image_pyramid[0].isContinuous() ? flat : flat.clone();
		memcpy(orig_tex.data(), img_array.data(), orig_tex.size());
		gli::sampler2d<float> Sampler_single(orig_tex, gli::WRAP_CLAMP_TO_EDGE, gli::FILTER_LINEAR, gli::FILTER_LINEAR, gli::vec4(1.0f, 0.5f, 0.0f, 1.0f));
		gli::texture2d newTex(orig_tex.format(), orig_tex.extent(), numMipmaps);
		memcpy(newTex.data(), orig_tex.data(), orig_tex.size());

//		gli::extent2d cood(512, 384);
//		gli::vec4 Sampletest = Sampler_single.texel_fetch(cood, 0);
//		cout << "\n============Sampletest val(RGBA):\n"
//			 << Sampletest.b << "," << Sampletest.g << "," << Sampletest.r << "," << Sampletest.a << endl;

                for (gli::texture::size_type level = 1; level < newTex.levels(); level++)
		{

			cv::Mat flat = image_pyramid[level].reshape(1, image_pyramid[level].total() * image_pyramid[level].channels());
			std::vector<float> img_array = image_pyramid[level].isContinuous() ? flat : flat.clone();
			memcpy(newTex.data(0, 0, level), img_array.data(), newTex.size(level));
			//			std::cout <<"\n show newTex.levels():"<<newTex.levels()<<endl;
			//			std::cout <<"\n show level:"<<level<<endl;
			//			std::cout <<"And dimension: newTex.extent(level).x :" <<newTex.extent(level).x << std::endl;
			//			std::cout <<"newTex.extent(level).y :" <<newTex.extent(level).y << std::endl;
		}


		static gli::sampler2d<float> Sampler(newTex, gli::WRAP_CLAMP_TO_EDGE, gli::FILTER_LINEAR, gli::FILTER_LINEAR, gli::vec4(1.0f, 0.5f, 0.0f, 1.0f));


//                prefilteredEnvmapSampler= &Sampler;


                gli::vec4 SampleA = Sampler.texture_lod(gli::fsampler2D::normalized_type(0.5f, 0.75f), 0.0f); // transform the texture coordinate
                cout << "\n============SampleA val------------------------(RGBA):\n"
			 << SampleA.b << "," << SampleA.g << "," << SampleA.r << "," << SampleA.a << endl;

//                gli::vec4 SampleAAAAAA = prefilteredEnvmapSampler->texture_lod(gli::fsampler2D::normalized_type(0.5f, 0.75f), 0.0f); // transform the texture coordinate
//                cout << "\n============SampleAAAAAA val(RGBA):\n"
//                     << SampleAAAAAA.b << "," << SampleAAAAAA.g << "," << SampleAAAAAA.r << "," << SampleAAAAAA.a << endl;



                return &Sampler;


	}




//	void EnvMapLookup::prefilteredColor(float u, float v, float level)
//	{
//          gli::vec4 Sample_val = prefilteredEnvmapSampler->texture_lod(gli::fsampler2D::normalized_type(0.5f, 0.75f),0.0f); // transform the texture coordinate
//
//          cout << "\n============Sample_val val(RGBA):\n" << Sample_val.b << "," << Sample_val.g << "," << Sample_val.r << ","   << Sample_val.a << endl;
////          return Vec3f(Sample_val.r, Sample_val.g, Sample_val.b);
//
//	}

	class brdfIntegrationMap
	{

	public:
		brdfIntegrationMap();
		~brdfIntegrationMap();
		gli::vec4 get_brdfIntegrationMap(float &NoV, float &roughness);
		Mat brdfIntegration_Map;

	private:
	};

	brdfIntegrationMap::brdfIntegrationMap()
	{

		string brdfIntegrationMap_path = "../include/brdfIntegrationMap/brdfIntegrationMap.pfm";
		brdfIntegration_Map = loadPFM(brdfIntegrationMap_path);
		cout << "show brdfIntegrationMap info:" << brdfIntegration_Map.depth() << "\nshow channel:" << brdfIntegration_Map.channels() << "\nshow width:" << brdfIntegration_Map.cols << "\nshow height:" << brdfIntegration_Map.rows << endl;
	}

	gli::vec4 brdfIntegrationMap::get_brdfIntegrationMap(float &NoV, float &roughness)
	{

		gli::texture2d brdf_tex(gli::FORMAT_RGB32_SFLOAT_PACK32, gli::texture2d::extent_type(128, 128), 1);

		cv::Mat flat_brdf = brdfIntegration_Map.reshape(1, brdfIntegration_Map.total() * brdfIntegration_Map.channels());
		std::vector<float> img_array = brdfIntegration_Map.isContinuous() ? flat_brdf : flat_brdf.clone();

		memcpy(brdf_tex.data(), img_array.data(), brdf_tex.size());
		//		vec4 brdfIntegration = texture(brdfIntegrationMap, vec2(NoV, roughness));
		gli::sampler2d<float> Sampler_brdf(brdf_tex, gli::WRAP_CLAMP_TO_EDGE, gli::FILTER_LINEAR, gli::FILTER_LINEAR, gli::vec4(1.0f, 0.5f, 0.0f, 1.0f));

		gli::vec4 SampleBrdf = Sampler_brdf.texture_lod(gli::fsampler2D::normalized_type(NoV, roughness), 0.0f); // transform the texture coordinate
		cout << "\n============SampleBrdf val(RGBA):\n"
			 << SampleBrdf.b << "," << SampleBrdf.g << "," << SampleBrdf.r << "," << SampleBrdf.a << endl;

		return SampleBrdf;

		// case 1
		// 0.5, 0.5
		//  image_ref_path_PFM blue,  green and red channel value: [0, 0.0179544, 0.724845]
		// ============SampleBrdf val(RGBA):
		//0.7253,0.0186449,0,1
		// case 2
		// gsn : 0.5, 0.75
		//		image_ref_path_PFM blue  green and red channel value: [0, 0.00670501, 0.570426]
		// GLI: 0.5, 0.25
		//		============SampleBrdf val(RGBA):  0.573688,0.00701116,0,1
	}

	brdfIntegrationMap::~brdfIntegrationMap()
	{
	}

	class diffuseMapMask
	{
	public:
		diffuseMapMask();
		~diffuseMapMask();

		void Init(int argc, char **argv);
		cv::Mat diffuse_Map_Mask;
		gli::vec4 getDiffuseMask(float &u, float &v);

	private:
	};

	diffuseMapMask::diffuseMapMask()
	{
	}

	diffuseMapMask::~diffuseMapMask()
	{
	}

	void diffuseMapMask::Init(int argc, char **argv)
	{

		render_diffuse_mask = new diffuse_Mask_Renderer;
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
		glutInitWindowPosition(100, 100);
		glutInitWindowSize(256, 128);

		window_diffuse_mask = glutCreateWindow("GSN Composer Shader Diffuse Mask Export");
		glutHideWindow();

		GLenum err = glewInit();
		if (GLEW_OK != err)
		{
			fprintf(stderr, "Glew error: %s\n", glewGetErrorString(err));
		}
		glutDisplayFunc(glutDisplay_diffuse_mask);
		glutReshapeFunc(glutResize_diffuse_mask);
		glutCloseFunc(glutClose_diffuse_mask);
		glutKeyboardFunc(glutKeyboard_diffuse_mask);

		render_diffuse_mask->init();
		glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
		glutTimerFunc(unsigned(20), timer_diffuse_mask, 0);
		glutMainLoop();
		diffuse_Map_Mask = img_diffuseMapMask;
	}

	gli::vec4 diffuseMapMask::getDiffuseMask(float &u, float &v)
	{
		// show the min max val
		double min_depth_val, max_depth_val;
		cv::minMaxLoc(diffuse_Map_Mask, &min_depth_val, &max_depth_val);
		cout << "\n show  diffuse_Map_Mask min, max:\n"
			 << min_depth_val << "," << max_depth_val << endl; // !!!!!!!!!!!!!!!!!!
		gli::texture2d diffuse_mask(gli::FORMAT_RGB32_SFLOAT_PACK32, gli::texture2d::extent_type(256, 128), 1);

		cv::Mat flat_diffuse = diffuse_Map_Mask.reshape(1, diffuse_Map_Mask.total() * diffuse_Map_Mask.channels());
		std::vector<float> img_diffuse_mask = diffuse_Map_Mask.isContinuous() ? flat_diffuse : flat_diffuse.clone();
		memcpy(diffuse_mask.data(), img_diffuse_mask.data(), diffuse_mask.size());

		gli::sampler2d<float> Sampler_diffuse_mask(diffuse_mask, gli::WRAP_CLAMP_TO_EDGE, gli::FILTER_LINEAR, gli::FILTER_LINEAR, gli::vec4(1.0f, 0.5f, 0.0f, 1.0f));
		gli::vec4 SampleDiffuseMask = Sampler_diffuse_mask.texture_lod(gli::fsampler2D::normalized_type(u, v), 0.0f);
		cout << "\n============SampleDiffuse val(RGBA):\n"
			 << SampleDiffuseMask.b << "," << SampleDiffuseMask.g << "," << SampleDiffuseMask.r << endl;

		return SampleDiffuseMask;
	}

	class diffuseMap
	{
	public:
		diffuseMap();
		~diffuseMap();

		void Init(int argc, char **argv);
		cv::Mat diffuse_Map;
		cv::Mat diffuse_Final_Map;
		void makeDiffuseMap();

		gli::vec4 getDiffuse(float &u, float &v);

	private:
	};

	diffuseMap::diffuseMap() {}
	diffuseMap::~diffuseMap() {}
	void diffuseMap::Init(int argc, char **argv)
	{
		render_diffuse = new Renderer_diffuse;
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
		glutInitWindowPosition(100, 100);
		glutInitWindowSize(256, 128);

		window_diffuse = glutCreateWindow("GSN Composer Shader Diffuse Export");
		glutHideWindow();
		GLenum err = glewInit();
		if (GLEW_OK != err)
		{
			fprintf(stderr, "Glew error: %s\n", glewGetErrorString(err));
		}
		glutDisplayFunc(glutDisplay_diffuse);
		glutReshapeFunc(glutResize_diffuse);
		glutCloseFunc(glutClose_diffuse);
		glutKeyboardFunc(glutKeyboard_diffuse);

		render_diffuse->init();
		glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
		glutTimerFunc(unsigned(20), timer_diffuse, 0);
		glutMainLoop();
		diffuse_Map = img_diffuseMap;
	}
	gli::vec4 diffuseMap::getDiffuse(float &u, float &v)
	{
		// show the min max val
		double min_depth_val, max_depth_val;
		cv::minMaxLoc(diffuse_Map, &min_depth_val, &max_depth_val);
		cout << "\n show  diffuse_Map min, max:\n"
			 << min_depth_val << "," << max_depth_val << endl; // !!!!!!!!!!!!!!!!!!
		gli::texture2d diffuse_tex(gli::FORMAT_RGB32_SFLOAT_PACK32, gli::texture2d::extent_type(256, 128), 1);

		cv::Mat flat_diffuse = diffuse_Map.reshape(1, diffuse_Map.total() * diffuse_Map.channels());
		std::vector<float> img_array_diffuse = diffuse_Map.isContinuous() ? flat_diffuse : flat_diffuse.clone();
		memcpy(diffuse_tex.data(), img_array_diffuse.data(), diffuse_tex.size());

		gli::sampler2d<float> Sampler_diffuse(diffuse_tex, gli::WRAP_CLAMP_TO_EDGE, gli::FILTER_LINEAR, gli::FILTER_LINEAR, gli::vec4(1.0f, 0.5f, 0.0f, 1.0f)); // TODO: check 1.0, 0.5 meaning
		gli::vec4 SampleDiffuse = Sampler_diffuse.texture_lod(gli::fsampler2D::normalized_type(u, v), 0.0f);
		cout << "\n============SampleDiffuse val(RGBA):\n"
			 << SampleDiffuse.b << "," << SampleDiffuse.g << "," << SampleDiffuse.r << "," << SampleDiffuse.a << endl;

		return SampleDiffuse;
	}

	Mat applyMask(Mat &input, Mat &mask)
	{

		for (int u = 0; u < input.rows; u++) // colId, cols: 0 to 128
		{
			for (int v = 0; v < input.cols; v++) // rowId,  rows: 0 to 256
			{

				if (mask.at<float>(u, v) == 0.0)
				{
					continue;
				}
				input.at<float>(u, v) = 1.0 / mask.at<float>(u, v);
			}
		}

		return input;
	}

	void makeDiffuseMap(Mat &diffuse_Map, Mat &diffuse_Mask, Mat &final_diffuseMap)
	{

		//	imshow("Diffuse Mask:", diffuse_Map);
		//	imshow("Diffuse Map:",  diffuse_Mask);

		Mat map_channel[3], mask_channel[3];

		split(diffuse_Map, map_channel);
		split(diffuse_Mask, mask_channel);
		std::vector<Mat> final_channels;

		final_channels.push_back(applyMask(map_channel[0], mask_channel[0]));
		final_channels.push_back(applyMask(map_channel[1], mask_channel[1]));
		final_channels.push_back(applyMask(map_channel[2], mask_channel[2]));

		merge(final_channels, final_diffuseMap);

		imshow("final_diffuseMap:", final_diffuseMap);
	}

}

// ======================convert the BGR to RGB==============================
//	cv::cvtColor(image_pyramid[0], image_pyramid[0], COLOR_BGR2RGB);

//  experiment notes:

//		gli::extent2d cood(512,384);
//		gli::vec4 Sampletest= Sampler_single.texel_fetch(cood,0);
//		cout<<"\n============Sampletest val(RGBA):\n"<<Sampletest.b<<","<<Sampletest.g<<","<<Sampletest.r<<","<<Sampletest.a<< endl;

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
//image_ref_path_PFM blue  green and red channel value:
// [0.0387254, 0.142647, 0.312745]

// textureLod 0.5, 0.25, 0;
//		============SampleA val(RGBA):
//		0.0421569,0.0254902,0.00833333,1

//case 3

//		gli::extent2d cood(512,384);(col, row) coordinate
//		============Sampletest val(RGBA):
//		0.305882,0.141176,0.0392157,1

// gsn composer  0.5,0.25,0:
//image_ref_path_PFM blue  green and red channel value:
// [0.0387254, 0.142647, 0.312745]

// textureLod 0.5, 0.75, 0;
//		============SampleA val(RGBA):
//		0.318627,0.145588,0.0387255,1

// =========================save data=====================================

//		std::string outputFile = "outputFile.dds";
//		gli::save(newTex, outputFile);
//		glm::highp_float32 TexelA = gli::textureLod<glm::highp_float32>(newTex, gli::texture2D::texcoord_type(0.0f, 0.0f), 0);

// =========================================================note==========================================

//		// checking
//		cv::Mat checkMemcpy(512, 1024, CV_32FC3);
//		memcpy(checkMemcpy.data, orig_tex.data(), orig_tex.size());
//		//		cv::cvtColor(checkMemcpy, checkMemcpy, COLOR_RGB2BGR) ;
//		imshow("checkMemcpy", checkMemcpy);
//
//		waitKey(0);

//		gli::texture2d  env_map_texture(tex.format(), tex.extent(), numMipmaps);
//
//
////		std::vector<gli::image2D> pyramid_Images;
//
//		for (int i = 0; i < 6; ++i) {
//
//			cv::cvtColor(image_pyramid[i], image_pyramid[i], COLOR_BGR2RGB) ;
//			cv::Mat flat = image_pyramid[i].reshape(1, image_pyramid[i].total() * image_pyramid[i].channels());
//			std::vector<float> img_array = image_pyramid[i].isContinuous() ? flat : flat.clone();
//			glm::uvec2 dim(image_pyramid[i].cols,image_pyramid[i].cols); // width, height
//
//			gli::texture2d env_map_texture();
////			gli::image2D  image_lvl_i(dim, gli::RGB32F, img_array);
////			pyramid_Images.push_back(image_lvl_i);
//
//		}
//
//
////		gli::texture2d tex(format, gli::texture2d::extent_type(Width, Height), 1);
////		memcpy(tex.data(), FreeImage_GetBits(bmp), tex.size());
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
