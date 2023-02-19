//
// Created by lei on 14.01.23.
//

#include "preFilter.h"
#include <boost/filesystem.hpp>
namespace DSONL {


    preFilterSpecularMask::preFilterSpecularMask(float &roughness) {

        _roughness = roughness;
        string shader_txt = "../include/shaders/specular_mask_fragment_shader.txt";

        fstream setRoughness(shader_txt);
        std::ostringstream oss;
        oss << std::setprecision(1) << _roughness;
        string roughness_str;
        (_roughness == 1 || _roughness == 0) ? roughness_str = oss.str() + ".0" : roughness_str = oss.str();

        float envmapLvl;
        roughness == (float) 0.2 ? envmapLvl = 0.0 : envmapLvl = roughness * 5.0;
        std::ostringstream oss_envl;
        oss_envl << std::setprecision(2) << envmapLvl;
        string envmapLvl_str = oss_envl.str() + ".0";

        setRoughness.seekg(221L, ios::beg);
        setRoughness << roughness_str;
        setRoughness.seekg(378L, ios::beg);
        setRoughness << envmapLvl_str;
        setRoughness.close();
    }

    preFilterSpecularMask::~preFilterSpecularMask() {

    }

    void preFilterSpecularMask::Init(int argc, char **argv,string parameter_path) {

        renderer_specular = new specular_Mask_Renderer(parameter_path);

        renderer_specular->roughness = _roughness;
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowPosition(100, 100);
        glutInitWindowSize(1024, 512);
        window_specular_mask = glutCreateWindow("GSN Composer Shader Mask Export");

        glutHideWindow();
        GLenum err = glewInit();
        if (GLEW_OK != err) { fprintf(stderr, "Glew error: %s\n", glewGetErrorString(err)); }
        glutDisplayFunc(glutDisplay_specular_mask);
        glutReshapeFunc(glutResize_specular_mask);
        glutCloseFunc(glutClose_specular_mask);
        glutKeyboardFunc(glutKeyboard_specular);

        // CodeOpt: use switch case here
        if (_roughness == (float) 0.0) {
            renderer_specular->resize(1024, 512);
        } else if (_roughness == (float) 0.2) {
            renderer_specular->resize(512, 256);
        } else if (_roughness == (float) 0.4) {
            renderer_specular->resize(256, 128);
        } else if (_roughness == (float) 0.6) {
            renderer_specular->resize(128, 64);
        } else if (_roughness == (float) 0.8) {
            renderer_specular->resize(64, 32);
        } else if (_roughness == (float) 1.0) {
            renderer_specular->resize(32, 16);
        } else {
            std::cerr << "Wrong roughness!" << endl;
        }
        renderer_specular->init();
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
        glutTimerFunc(unsigned(20), timer_specular, 0);
        glutMainLoop();
    }



    preFilter::preFilter(float &roughness) {

        //          _roughness = roughness;
        //          float envmapLvl;
        //          std::ostringstream oss, oss_envl;
        //          oss << std::setprecision(1) << _roughness;
        //          string roughness_str, envmapLvl_str;
        //          (_roughness == 1 || _roughness == 0) ? roughness_str = oss.str() +
        //          ".0" : roughness_str = oss.str(); roughness == (float)0.2 ?
        //          envmapLvl = 0.0 : envmapLvl = roughness * 5.0; oss_envl <<
        //          std::setprecision(2) << envmapLvl; envmapLvl_str = oss_envl.str()
        //          + ".0"; modify_shader_file(roughness_str, envmapLvl_str);

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
        roughness == (float) 0.2 ? envmapLvl = 0.0 : envmapLvl = roughness * 5.0;
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

    void preFilter::Init(int argc, char **argv, string parameter_path) {
        renderer = new Renderer(parameter_path);
        renderer->roughness = _roughness;
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowPosition(100, 100);
        glutInitWindowSize(1024, 512);
        window = glutCreateWindow("GSN Composer Shader Export");
        glutHideWindow();
        GLenum err = glewInit();
        if (GLEW_OK != err) { fprintf(stderr, "Glew error: %s\n", glewGetErrorString(err)); }
        glutDisplayFunc(glutDisplay);
        glutReshapeFunc(glutResize);
        glutCloseFunc(glutClose);
        glutKeyboardFunc(glutKeyboard);

        if (_roughness == (float) 0.0) {
            renderer->resize(1024, 512);
        } else if (_roughness == (float) 0.2) {
            renderer->resize(512, 256);
        } else if (_roughness == (float) 0.4) {
            renderer->resize(256, 128);
        } else if (_roughness == (float) 0.6) {
            renderer->resize(128, 64);
        } else if (_roughness == (float) 0.8) {
            renderer->resize(64, 32);
        } else if (_roughness == (float) 1.0) {
            renderer->resize(32, 16);
        } else {
            std::cerr << "Wrong roughness!" << endl;
        }
        renderer->init();
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
        glutTimerFunc(unsigned(20), timer, 0);
        glutMainLoop();
    }


    EnvMapLookup::EnvMapLookup(int argc, char **argv, string parameter_path, string lightIdx) {

        float roughness_i = 0.0;
        for (int i = 0; i < 6; ++i) {
            roughness_i = i / 5.0;
            preFilterSpecularMask env_map_preFilterMask(roughness_i);
//            string parameter_path_envmask="include/preFilter_data/parameters_envmapMask.csv";
//			env_map_preFilterMask.Init(argc, argv, parameter_path_envmask);
            env_map_preFilterMask.Init(argc, argv, parameter_path);
        }

        std::vector<cv::Mat> image_pyramid_mask_mid(img_pyramid_mask);



        img_pyramid_mask.clear();
        image_pyramid_mask = image_pyramid_mask_mid;



        roughness_i = 0.0;
        for (int i = 0; i < 6; ++i) {
            roughness_i = i / 5.0;
            preFilter env_mpa_preFilter(roughness_i);
            env_mpa_preFilter.Init(argc, argv, parameter_path);
        }

        std::vector<cv::Mat> image_pyramid_middle(img_pyramid);
        img_pyramid.clear();
        image_pyramid = image_pyramid_middle;

        mergeImagePyramid( lightIdx);

        //                double min_depth_val, max_depth_val;
        //                cv::minMaxLoc(image_pyramid[0], &min_depth_val,&max_depth_val);
        //                cout<<"show original image_pyramid range:"<<min_depth_val<<","<<max_depth_val<<endl;
    }


    EnvMapLookup::~EnvMapLookup() {}

    cv::Mat EnvMapLookup::applyMask(Mat &input, Mat &mask) {

        for (int u = 0; u < input.rows; u++) {
            for (int v = 0; v < input.cols; v++) {
                if (mask.at<float>(u, v) == 0.0f) { continue; }
                input.at<float>(u, v) = std::exp(1.0 / mask.at<float>(u, v) - 1.0f);
            }
        }
        return input;
    }


    void EnvMapLookup::mergeImagePyramid( string lightIdx) {

//        string renderedEnvLight_path="/home/lei/Documents/Research/envMapData/envMapData_Dense01";
//        string renderedEnvLight_path="/home/lei/Documents/Research/envMapData/maskedSelector";
//        string renderedEnvLight_path="/home/lei/Documents/Research/envMapData/SeventeenPointsEnvMap";
//        string renderedEnvLight_path="/home/lei/Documents/Research/envMapData/EnvMap_764";

//        string renderedEnvLight_path= "/home/lei/Documents/Research/envMapData/EnvMap_91";
        string renderedEnvLight_path= "/home/lei/Documents/Research/envMapData/EnvMap150_wholeImg";






        string renderedEnvLightfolder= renderedEnvLight_path+ "/envMap"+lightIdx+"/renderedEnvLight";  //+"/renderedEnvLight";


        if (boost::filesystem::create_directories(renderedEnvLightfolder)){
            cout << "Directory:"+ renderedEnvLightfolder +" created"<<endl;
        }




        for (int i = 0; i < 6; ++i) {
            Mat map_channel[3], mask_channel[3];
            split(image_pyramid[i], map_channel);
            split(image_pyramid_mask[i], mask_channel);
            std::vector<Mat> final_channels;
            final_channels.push_back(applyMask(map_channel[0], mask_channel[0]));
            final_channels.push_back(applyMask(map_channel[1], mask_channel[1]));
            final_channels.push_back(applyMask(map_channel[2], mask_channel[2]));
            cv::merge(final_channels, image_pyramid[i]);

            // save image_pyramid
            stringstream ss;
            string img_idx_str;
            ss << i;
            ss >> img_idx_str;
            string name_prefix = "/envMapLvl_";
            string image_pyramid_lvl= renderedEnvLightfolder+name_prefix+img_idx_str +".pfm";
            savePFM(image_pyramid[i], image_pyramid_lvl);

        }



    }

    void EnvMapLookup::makeMipMap(std::vector<gli::sampler2d<float>>& Sampler_vec) {

        //===============================mind  opencv BGR order=================================
        // define and allocate space for Mipmap using GLM library

        std::size_t numMipmaps = 6;
        gli::texture2d orig_tex(gli::FORMAT_RGB32_SFLOAT_PACK32, gli::texture2d::extent_type(1024, 512), 1);
        cv::Mat flat = image_pyramid[0].reshape(1, image_pyramid[0].total() * image_pyramid[0].channels());
        std::vector<float> img_array = image_pyramid[0].isContinuous() ? flat : flat.clone();
        memcpy(orig_tex.data(), img_array.data(), orig_tex.size());
        gli::sampler2d<float> Sampler_single(orig_tex, gli::WRAP_CLAMP_TO_EDGE, gli::FILTER_LINEAR, gli::FILTER_LINEAR, gli::vec4(1.0f, 0.5f, 0.0f, 1.0f));
        gli::texture2d newTex(orig_tex.format(), orig_tex.extent(), numMipmaps);
        memcpy(newTex.data(), orig_tex.data(), orig_tex.size());

        for (gli::texture::size_type level = 1; level < newTex.levels(); level++) {
            cv::Mat flat = image_pyramid[level].reshape(1, image_pyramid[level].total() * image_pyramid[level].channels());
            std::vector<float> img_array = image_pyramid[level].isContinuous() ? flat : flat.clone();
            memcpy(newTex.data(0, 0, level), img_array.data(), newTex.size(level));
        }



//		static gli::sampler2d<float> Sampler(newTex, gli::WRAP_CLAMP_TO_EDGE, gli::FILTER_LINEAR, gli::FILTER_LINEAR, gli::vec4(1.0f, 0.5f, 0.0f, 1.0f));
//		prefilteredEnvmapSampler = &Sampler;

        gli::sampler2d<float> Sampler(newTex, gli::WRAP_CLAMP_TO_EDGE, gli::FILTER_LINEAR, gli::FILTER_LINEAR, gli::vec4(1.0f, 0.5f, 0.0f, 1.0f));


//        gli::vec4 SampleSpecular0 = Sampler.texture_lod(gli::fsampler2D::normalized_type(0.939755, 1.0-0.722123), 0.5* 5.0);

//            [0.151013, 0.074238, 0.0679612]
//        std::cout << "\n=====INSIDE start=======SampleSpecular val(BGRA):" << SampleSpecular0.b << "," << SampleSpecular0.g << "," << SampleSpecular0.r << ","   << SampleSpecular0.a << std:: endl;


        Sampler_vec.push_back(Sampler);

//        gli::vec4 SampleSpecular = Sampler_vec[0].texture_lod(gli::fsampler2D::normalized_type(0.939755, 1.0-0.722123), 0.5 * 5.0);

//            [0.151013, 0.074238, 0.0679612]
//        std::cout << "\n=====INSIDE=======SampleSpecular val(BGRA):" << SampleSpecular.b << "," << SampleSpecular.g << "," << SampleSpecular.r << ","   << SampleSpecular.a << std:: endl;


//                gli::vec4 SampleA = Sampler.texture_lod(gli::fsampler2D::normalized_type(0.5f,0.75f), 0.0f); // transform the texture coordinate
//                cout << "\n============SampleA val------------------------(RGBA):\n" << SampleA.b << "," << SampleA.g << "," << SampleA.r << "," <<SampleA.a << endl;
        //                gli::vec4 SampleA = Sampler.texture_lod(gli::fsampler2D::normalized_type(0.5f,0.75f), 0.0f); // transform the texture coordinate
        //                cout << "\n============SampleA val------------------------(RGBA):\n" << SampleA.b << "," << SampleA.g << "," << SampleA.r << "," <<SampleA.a << endl;
        //                gli::vec4 SampleAAAAAA =prefilteredEnvmapSampler->texture_lod(gli::fsampler2D::normalized_type(0.5f,0.75f), 0.0f); // transform the texture coordinate
        //                cout << "\n============SampleAAAAAA val(RGBA):\n"<< SampleAAAAAA.b << "," << SampleAAAAAA.g << "," <<SampleAAAAAA.r << "," << SampleAAAAAA.a << endl;
    }



    brdfIntegrationMap::brdfIntegrationMap(string map_path) {

        brdf_path=map_path;

//		string brdfIntegrationMap_path = "../include/brdfIntegrationMap/brdfIntegrationMap.pfm";
        brdfIntegration_Map = loadPFM(brdf_path);

    }

    void brdfIntegrationMap::makebrdfIntegrationMap( std::vector<gli::sampler2d<float>> &brdf ) {

        gli::texture2d brdf_tex(gli::FORMAT_RGB32_SFLOAT_PACK32, gli::texture2d::extent_type(128, 128), 1);
        cv::Mat flat_brdf = brdfIntegration_Map.reshape(1, brdfIntegration_Map.total() * brdfIntegration_Map.channels());
        std::vector<float> img_array = brdfIntegration_Map.isContinuous() ? flat_brdf : flat_brdf.clone();
        memcpy(brdf_tex.data(), img_array.data(), brdf_tex.size());
//		static gli::sampler2d<float> Sampler_brdf(brdf_tex, gli::WRAP_CLAMP_TO_EDGE, gli::FILTER_LINEAR, gli::FILTER_LINEAR, gli::vec4(1.0f, 0.5f, 0.0f, 1.0f));
//		brdfSampler = &Sampler_brdf;

        gli::sampler2d<float> Sampler_brdf(brdf_tex, gli::WRAP_CLAMP_TO_EDGE, gli::FILTER_LINEAR, gli::FILTER_LINEAR, gli::vec4(1.0f, 0.5f, 0.0f, 1.0f));
        brdf.push_back(Sampler_brdf);





    }

    brdfIntegrationMap::~brdfIntegrationMap() {}



    diffuseMapMask::diffuseMapMask() {}

    diffuseMapMask::~diffuseMapMask() {}

    void diffuseMapMask::Init(int argc, char **argv,string parameters_path) {
        int diffuseMapMaskWidth = 256;
        int diffuseMapMaskHeight = 128;
        render_diffuse_mask = new diffuse_Mask_Renderer( parameters_path);
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowPosition(100, 100);
        glutInitWindowSize(diffuseMapMaskWidth, diffuseMapMaskHeight);

        window_diffuse_mask = glutCreateWindow("GSN Composer Shader Diffuse Mask Export");
        glutHideWindow();

        GLenum err = glewInit();
        if (GLEW_OK != err) { fprintf(stderr, "Glew error: %s\n", glewGetErrorString(err)); }
        glutDisplayFunc(glutDisplay_diffuse_mask);
        glutReshapeFunc(glutResize_diffuse_mask);
        glutCloseFunc(glutClose_diffuse_mask);
        glutKeyboardFunc(glutKeyboard_diffuse_mask);

        render_diffuse_mask->resize(diffuseMapMaskWidth, diffuseMapMaskHeight);

        render_diffuse_mask->init();
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
        glutTimerFunc(unsigned(20), timer_diffuse_mask, 0);
        glutMainLoop();
        diffuse_Map_Mask = img_diffuseMapMask;

        //                delete render_diffuse_mask;
        //                render_diffuse_mask=NULL;
    }

    void diffuseMapMask::makeDiffuseMask() {
        //		// show the min max val
        //		double min_depth_val, max_depth_val;
        //		cv::minMaxLoc(diffuse_Map_Mask, &min_depth_val, &max_depth_val);
        //		cout << "\n show  diffuse_Map_Mask min, max:\n"<< min_depth_val <<
        //"," << max_depth_val << endl; // !!!!!!!!!!!!!!!!!!
        gli::texture2d diffuse_mask(gli::FORMAT_RGB32_SFLOAT_PACK32, gli::texture2d::extent_type(256, 128), 1);
        cv::Mat flat_diffuse = diffuse_Map_Mask.reshape(1, diffuse_Map_Mask.total() * diffuse_Map_Mask.channels());
        std::vector<float> img_diffuse_mask = diffuse_Map_Mask.isContinuous() ? flat_diffuse : flat_diffuse.clone();
        memcpy(diffuse_mask.data(), img_diffuse_mask.data(), diffuse_mask.size());
        gli::sampler2d<float> Sampler_diffuse_mask(diffuse_mask, gli::WRAP_CLAMP_TO_EDGE, gli::FILTER_LINEAR, gli::FILTER_LINEAR, gli::vec4(1.0f, 0.5f, 0.0f, 1.0f));
        //		gli::vec4 SampleDiffuseMask =
        //Sampler_diffuse_mask.texture_lod(gli::fsampler2D::normalized_type(0.5,
        //0.25), 0.0f);
    }




    diffuseMap::diffuseMap() {}
    diffuseMap::~diffuseMap() {}
    void diffuseMap::Init(int argc, char **argv, string parameters_path, string img_idx_str) {

        int diffuseMapWidth = 256;
        int diffuseMapHeight = 128;

        render_diffuse = new Renderer_diffuse(parameters_path);
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowPosition(100, 100);
        glutInitWindowSize(diffuseMapWidth, diffuseMapHeight);

        window_diffuse = glutCreateWindow("GSN Composer Shader Diffuse Export");
        glutHideWindow();
        GLenum err = glewInit();
        if (GLEW_OK != err) { fprintf(stderr, "Glew error: %s\n", glewGetErrorString(err)); }
        glutDisplayFunc(glutDisplay_diffuse);
        glutReshapeFunc(glutResize_diffuse);
        glutCloseFunc(glutClose_diffuse);
        glutKeyboardFunc(glutKeyboard_diffuse);

        render_diffuse->resize(diffuseMapWidth, diffuseMapHeight);

        render_diffuse->init();
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
        glutTimerFunc(unsigned(20), timer_diffuse, 0);
        glutMainLoop();

        diffuse_Map = img_diffuseMap;
        // get diffuse mask map
        diffuseMapMask *DiffuseMaskMap = new diffuseMapMask;
        DiffuseMaskMap->Init(argc, argv,parameters_path);
        // DiffuseMaskMap->makeDiffuseMask();
        diffuse_MaskMap = DiffuseMaskMap->diffuse_Map_Mask;
        //                imshow("diffuse_Map",diffuse_Map);
        //                imshow("diffuse_MaskMap",diffuse_MaskMap);
        //                waitKey(0);
        delete DiffuseMaskMap;
        DiffuseMaskMap = NULL;
    }
    gli::vec4 diffuseMap::getDiffuse(float u, float v) {
        // show the min max val
        double min_depth_val, max_depth_val;
        cv::minMaxLoc(diffuse_Map, &min_depth_val, &max_depth_val);
        cout << "\n show  diffuse_Map min, max:\n" << min_depth_val << "," << max_depth_val << endl;// !!!!!!!!!!!!!!!!!!
        gli::texture2d diffuse_tex(gli::FORMAT_RGB32_SFLOAT_PACK32, gli::texture2d::extent_type(256, 128), 1);

        cv::Mat flat_diffuse = diffuse_Map.reshape(1, diffuse_Map.total() * diffuse_Map.channels());
        std::vector<float> img_array_diffuse = diffuse_Map.isContinuous() ? flat_diffuse : flat_diffuse.clone();
        memcpy(diffuse_tex.data(), img_array_diffuse.data(), diffuse_tex.size());

        gli::sampler2d<float> Sampler_diffuse(diffuse_tex, gli::WRAP_CLAMP_TO_EDGE, gli::FILTER_LINEAR, gli::FILTER_LINEAR, gli::vec4(1.0f, 0.5f, 0.0f, 1.0f));// TODO: check 1.0, 0.5 meaning
        gli::vec4 SampleDiffuse = Sampler_diffuse.texture_lod(gli::fsampler2D::normalized_type(u, v), 0.0f);
        cout << "\n============SampleDiffuse val(RGBA):\n" << SampleDiffuse.b << "," << SampleDiffuse.g << "," << SampleDiffuse.r << "," << SampleDiffuse.a << endl;

        gli::vec4 SampleDiffuseTemp = Sampler_diffuse.texture_lod(gli::fsampler2D::normalized_type(0.5, 0.25), 0.0f);
        cout << "\n============SampleDiffuse val(RGBA):\n" << SampleDiffuseTemp.b << "," << SampleDiffuseTemp.g << "," << SampleDiffuseTemp.r << "," << SampleDiffuseTemp.a << endl;

        return SampleDiffuse;
    }

    void diffuseMap::mergeDiffuse_Map( string lightIdx) {



//        string renderedEnvLight_path="/home/lei/Documents/Research/envMapData/envMapData_Dense01";
//        string renderedEnvLight_path="/home/lei/Documents/Research/envMapData/maskedSelector";
//        string renderedEnvLight_path="/home/lei/Documents/Research/envMapData/SeventeenPointsEnvMap";

//        string renderedEnvLight_path="/home/lei/Documents/Research/envMapData/EnvMap_764";
//        string renderedEnvLight_path="/home/lei/Documents/Research/envMapData/EnvMap_91";

        string renderedEnvLight_path="/home/lei/Documents/Research/envMapData/EnvMap150_wholeImg";





        string renderedEnvLightDiffuse= renderedEnvLight_path+ "/envMap"+lightIdx+"/renderedEnvLightDiffuse"; //+"/renderedEnvLight";
        if (boost::filesystem::create_directories(renderedEnvLightDiffuse)){
            cout << "Directory:"+ renderedEnvLightDiffuse +" created"<<endl;
        }



        Mat map_channel[3], mask_channel[3];
        split(diffuse_Map, map_channel);
        split(diffuse_MaskMap, mask_channel);
        std::vector<Mat> final_channels;
        final_channels.push_back(applyMask(map_channel[0], mask_channel[0]));
        final_channels.push_back(applyMask(map_channel[1], mask_channel[1]));
        final_channels.push_back(applyMask(map_channel[2], mask_channel[2]));
        cv::merge(final_channels, diffuse_Final_Map);


        // save merged Diffuse envMap

        stringstream ss;
        string img_idx_str;
        ss << lightIdx;
        ss >> img_idx_str;
        string name_prefix = "/envMapDiffuse_";
        string envMapDiffuse= renderedEnvLightDiffuse+name_prefix+img_idx_str +".pfm";
        savePFM(diffuse_Final_Map, envMapDiffuse);





        //	double min_depth_val, max_depth_val;
        //	cv::minMaxLoc(diffuse_Final_Map, &min_depth_val, &max_depth_val);
        //	cout<<"================"<<"min_depth_val"<<min_depth_val<<"max_depth_val"<<max_depth_val<<endl;
        //  imshow("diffuse_Final_Map",diffuse_Final_Map);
        //  waitKey(0);
        //          cout<<"====================check diffuse_Final_Map
        //          size============="<<diffuse_Final_Map.rows<<
        //              "and,"<<diffuse_Final_Map.cols<<endl;
    }
    cv::Mat diffuseMap::applyMask(Mat &input, Mat &mask) {
        for (int u = 0; u < input.rows; u++)// colId, cols: 0 to 128
        {
            for (int v = 0; v < input.cols; v++)// rowId,  rows: 0 to 256
            {

                if (mask.at<float>(u, v) == 0.0) { continue; }
                input.at<float>(u, v) = 1.0 / mask.at<float>(u, v);
            }
        }
        return input;
    }

    void diffuseMap::makeDiffuseMap( std::vector<gli::sampler2d<float>>& Sampler_diffuse_vec, string  img_idx_str) {
        mergeDiffuse_Map( img_idx_str );
//        gli::texture2d diffuse_tex(gli::FORMAT_RGB32_SFLOAT_PACK32, gli::texture2d::extent_type(256, 128), 1);
//        cv::Mat flat_diffuse = diffuse_Final_Map.reshape(1, diffuse_Final_Map.total() * diffuse_Final_Map.channels());
//        std::vector<float> img_array_diffuse = diffuse_Final_Map.isContinuous() ? flat_diffuse : flat_diffuse.clone();
//        memcpy(diffuse_tex.data(), img_array_diffuse.data(), diffuse_tex.size());
//
////		static gli::sampler2d<float> Sampler_diffuse(diffuse_tex, gli::WRAP_CLAMP_TO_EDGE, gli::FILTER_LINEAR, gli::FILTER_LINEAR, gli::vec4(1.0f, 0.5f, 0.0f, 1.0f));// TODO: check 1.0, 0.5 meaning
////
////		diffuseSampler = &Sampler_diffuse;
//
//
//        gli::sampler2d<float> Sampler_diffuse(diffuse_tex, gli::WRAP_CLAMP_TO_EDGE, gli::FILTER_LINEAR, gli::FILTER_LINEAR, gli::vec4(1.0f, 0.5f, 0.0f, 1.0f));// TODO: check 1.0, 0.5 meaning
//
//
//
//
//
//        Sampler_diffuse_vec.push_back(Sampler_diffuse);

        //          gli::vec4 SampleDiffuse =Sampler_diffuse.texture_lod(gli::fsampler2D::normalized_type(0.5, 1-0.75), 0.0f);
        //          cout << "\n============SampleDiffuse val(RGBA):\n" << SampleDiffuse.r << "," << SampleDiffuse.g << "," << SampleDiffuse.b << "," << SampleDiffuse.a << endl;
    }




}