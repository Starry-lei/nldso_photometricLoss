// Copyright (C) Thorsten Thormaehlen, MIT License (see license file)

#include <string>
#include <iostream> 
#include <iomanip>
#include <fstream>
#include <sstream>
#include <ctype.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include "Renderer.h"
#include "Mesh.h"
#include "ShaderNode.h"
#include "LoadOBJ.h"
#include "Matrix.h"
#include "FileTools.h"
#include "StringTools.h"

using namespace std;
using namespace gsn;

Renderer::Renderer()
{
  t = 0.0;
  windowWidth = 0;
  windowHeight = 0;
  selectedOutput = 0;
}

Renderer::~Renderer() {
}

void Renderer::init() {

  // Initialize shader A
  shaderSettingsA = loadShaderSettings(FileTools::findFile("data/parameters.csv"));

  if (shaderSettingsA.nodeClassName == "ImageShaderPluginNode") {
    // for image shaders, no mesh is required but a screen-aligned quad
    meshA.createQuad();
    // for image shaders, only the fragment shader code is required, 
    // the vertex shader is set to default
    std::string f = FileTools::findFile("shaders/fragment_shader.txt");
    shaderNodeA.setShaderSourceFromFile("", f);
  } else {
    // for regular shaders, load the input mesh
    LoadOBJ::load(FileTools::findFile("data/Mesh.obj"), meshA);
    // for regular shaders, a fragment and a vertex shader is required
    std::string v = FileTools::findFile("shaders/vertex_shader.txt");
    std::string f = FileTools::findFile("shaders/fragment_shader.txt");
    shaderNodeA.setShaderSourceFromFile(v, f);
  }
  
  shaderNodeA.setUniformsFromFile(FileTools::findFile("data/parameters.csv"));
  
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

void Renderer::resize(int w, int h) {
  windowWidth = w;
  windowHeight = h;
}

void Renderer::display() {

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
}

void Renderer::dispose() {
}

Renderer::ShaderSettings Renderer::loadShaderSettings(const std::string& filename) const
{
  Renderer::ShaderSettings settings;

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