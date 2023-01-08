// Copyright (C) Thorsten Thormaehlen, MIT License (see license file)

#include <GL/glew.h>
#include <string>
#include <iostream> 
#include <iomanip>
#include <fstream>
#include <sstream>
#include <ctype.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include "Renderer_diffuse.h"
#include "preFilter/Mesh.h"
#include "preFilter/ShaderNode.h"
#include "preFilter/LoadOBJ.h"
#include "preFilter/Matrix.h"
#include "preFilter/FileTools.h"
#include "preFilter/StringTools.h"

#include "settings/preComputeSetting.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace gsn;

Renderer_diffuse::Renderer_diffuse(std::string parameters_path)
{
  t = 0.0;
  windowWidth = 0;
  windowHeight = 0;
  selectedOutput = 0;
  parameters_path_=parameters_path;
}

Renderer_diffuse::~Renderer_diffuse() {
}

void Renderer_diffuse::init() {

    // parameters_path_= "include/diffuseMap/parameters.csv";
  // Initialize shader A
  shaderSettingsA = loadShaderSettings(FileTools::findFile(parameters_path_));

  if (shaderSettingsA.nodeClassName == "ImageShaderPluginNode") {
    // for image shaders, no mesh is required but a screen-aligned quad
    meshA.createQuad();
    // for image shaders, only the fragment shader code is required, 
    // the vertex shader is set to default
    std::string f = FileTools::findFile("include/shaders/diffuse_fragment_shader.txt");
    shaderNodeA.setShaderSourceFromFile("", f);
  } else {
//    // for regular shaders, load the input mesh
//    LoadOBJ::load(FileTools::findFile("data/Mesh.obj"), meshA);
//    // for regular shaders, a fragment and a vertex shader is required
//    std::string v = FileTools::findFile("shaders/vertex_shader.txt");
//    std::string f = FileTools::findFile("shaders/fragment_shader.txt");
//    shaderNodeA.setShaderSourceFromFile(v, f);
  }
  
//  shaderNodeA.setUniformsFromFile(FileTools::findFile("include/diffuseMap/parameters.csv"));
    shaderNodeA.setUniformsFromFile(FileTools::findFile(parameters_path_));
  
  // Initialize shader B
  // Shader B is an image shader that renders a screen-aligned quad
  // with the output of shader A as texture
  std::string fragSrc;
  fragSrc += "#version 300 es\n";
  fragSrc += "precision highp float;\n";
  fragSrc += "out vec4 outColor;\n";
  fragSrc += "in vec2 tc; // texture coordinate of the output image in range [0.0, 1.0]\n";
  fragSrc += "\n";
  fragSrc += "uniform sampler2D img; // mag_filter=\"LINEAR\" \n";
  fragSrc += "uniform float aspectX;\n";
  fragSrc += "uniform float aspectY;\n";
  fragSrc += "\n";
  fragSrc += "void main() {\n";
  fragSrc += "  vec2 tcc = (vec2(aspectX, aspectY) * (tc - vec2(0.5))) + vec2(0.5);\n";
  fragSrc += "  if(tcc.x >= 0.0 && tcc.x <= 1.0 && tcc.y >= 0.0 && tcc.y <= 1.0) {\n";
  fragSrc += "    outColor = texture(img, tcc);\n";
  fragSrc += "  } else {\n";
  fragSrc += "    discard;\n";
  fragSrc += "  }\n";
  fragSrc += "}\n";
  shaderNodeB.setShaderSource("", fragSrc);
  meshB.createQuad();
  shaderSettingsB.backgroundColor = Matrix(0.0, 0.0, 0.0, 1.0);

}

void Renderer_diffuse::resize(int w, int h) {
  windowWidth = w;
  windowHeight = h;
}

void Renderer_diffuse::display() {

  // You can manually overwrite the uniform variables, e.g.,
  //
  // Matrix pers(4, 4);
  // pers.setPerspective(45.0f, (float)windowWidth / (float)windowHeight, 0.5f, 40.0f);
  // shaderNodeA.setUniformMatrix("projection", pers);
  // cout << pers.prettyString() << endl;

  // Matrix lookat(4, 4);
  // float rad = float(M_PI) / 180.0f * t;
  // lookat.setLookAt(5.0f * float(cos(rad)), 5.0f, 5.0f * float(sin(rad)), // eye
  //                  0.0f, 0.0f, 0.0f, // look at
  //                  0.0f, 1.0f, 0.0f); // up
  // shaderNodeA.setUniformMatrix("cameraLookAt", lookat);


  // render meshA with shaderNodeA to FBO
  shaderNodeA.render(meshA, shaderSettingsA.backgroundColor, shaderSettingsA.width, shaderSettingsA.height, shaderSettingsA.wireframe, true);


  // compute aspect ratio for shaderNodeB
  float aspectX = 1.0;
  float aspectY = 1.0;
  float aspectShader = float(shaderSettingsA.width) / float(shaderSettingsA.height);
  float aspectWindow = float(windowWidth) / float(windowHeight);
  if (aspectShader >= aspectWindow) {
    aspectY = aspectShader / aspectWindow;
  } else {
    aspectX = aspectWindow / aspectShader;
  }
  shaderNodeB.setUniformFloat("aspectX", aspectX);
  shaderNodeB.setUniformFloat("aspectY", aspectY);

  // use output of shaderNodeA as uniform input for shaderNodeB
  shaderNodeB.setUniformImage("img", shaderNodeA.getRenderTarget(selectedOutput));

  // render meshB (a screen-aligned quad) with shaderNodeB to screen
  shaderNodeB.render(meshB, shaderSettingsB.backgroundColor, windowWidth, windowHeight, false, false);

	int height= windowHeight;
	int width= windowWidth;
	//	cv::Mat saveBufferMat(height, width, CV_8UC3);
	cv::Mat saveBufferMat(height, width, CV_32FC3);
//	cv::Mat saveBufferMat(height, width, CV_16UC3);
//	glPixelStorei(GL_PACK_ALIGNMENT, 4);
	glPixelStoref(GL_PACK_ALIGNMENT, 8);

//	float array[1024*512*3];

	glReadPixels(0,0,width,height,GL_BGR,GL_FLOAT,saveBufferMat.data);

//	glReadPixels(0,0,width,height,GL_BGR,GL_FLOAT,saveBufferMat.data);// GL_HALF_FLOAT
//	glReadPixels(0,0,width,height,GL_BGR,GL_FLOAT,saveBufferMat.data);
//	glReadPixels(0,0,width,height,GL_BGR,GL_UNSIGNED_BYTE,array);
//	glReadPixels(0,0,width,height,GL_BGR,GL_FLOAT,array);//GL_UNSIGNED_SHORT_5_6_5
//	glReadPixels(0,0,width,height,GL_BGR,GL_FLOAT,array);//GL_UNSIGNED_SHORT_5_6_5


//	cout<<"shwo array info:"<<array[1]<<endl;
// std::vector<float> vector(std::begin(array), std::end(array));
// float max = *max_element(vector.begin(), vector.end());
// cout<<"==========Max value:=============="<<max<<endl;
// float min= *min_element(vector.begin(), vector.end());
// cout<<"==========Min value:=============="<<min<<endl;
//	cout <<"\nwidth:"<<width<<"\nheight:"<<height<<endl;


	cv::Mat flipped;
	cv::flip(saveBufferMat, flipped, 0);

	DSONL::img_diffuseMap=flipped;

	//	cout<<"\n img_diffuseMap depth():"<<flipped.depth()<<"\n Image saved! And its width: "<< flipped.cols<<", its height:"<<flipped.rows<<endl;
	//	// show images
	//	cv::imshow("show flipped img_diffuseMap!",flipped);
}

void Renderer_diffuse::dispose() {
}

Renderer_diffuse::ShaderSettings Renderer_diffuse::loadShaderSettings(const std::string& filename) const
{
	Renderer_diffuse::ShaderSettings settings;
        settings.width=windowWidth;// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1attention!!!!!!!!!!!!!!!!!!!!!!!!
        settings.height=windowHeight;

  std::string content = FileTools::readTextFile(filename);
  std::vector <string> lines = StringTools::split(content, "\n");
  for (int i = 0; i < int(lines.size()); i++) {
    std::string& line = lines[i];
    std::vector <string> items = StringTools::split(line, ",");
    if (items.size() >= 2) {
      string name = items[0];
      string type = StringTools::toLower(items[1]);
      if (type == "integer" && name == "width"  && items.size() >= 3) {
        settings.width = StringTools::stringToInt(items[2]);
      }
      if (type == "integer" && name == "height" && items.size() >= 3) {
        settings.height = StringTools::stringToInt(items[2]);
      }
      if (type == "boolean" && name == "Wireframe" &&  items.size() >= 3) {
        settings.wireframe = StringTools::stringToBool(items[2]);
      }
      if (type == "color" && name == "Background" && items.size() >= 6) {
        float x = StringTools::stringToFloat(items[2]);
        float y = StringTools::stringToFloat(items[3]);
        float z = StringTools::stringToFloat(items[4]);
        float w = StringTools::stringToFloat(items[5]);
        Matrix mat(x, y, z, w);
        settings.backgroundColor = mat;
      }
      if (type == "text" && name == "NodeClassName" && items.size() >= 3) {
        settings.nodeClassName = items[2];
      }
      if (type == "text" && name == "NodeName" && items.size() >= 3) {
        settings.nodeName = items[2];
      }
    }
  }

  return settings;
}