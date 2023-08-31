// Copyright (C) Thorsten Thormaehlen, MIT License (see license file)

#ifndef DEF_DIFFUSE_MASK_RENDERER_H
#define DEF_DIFFUSE_MASK_RENDERER_H

#include <stdio.h>
#include <string>
#include <vector>
#include "preFilter/Mesh.h"
#include "preFilter/ShaderNode.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

namespace gsn {
  /*!
  \class Renderer Renderer.h
  \brief This class defines a renderer.
  */

  class diffuse_Mask_Renderer {
  public:

    //! constructor
    diffuse_Mask_Renderer(std::string parameters_path);

    //! destructor
    ~diffuse_Mask_Renderer();

  public:
    //! initialize all rendering resources
    void init();

    //! resize event
    void resize(int w, int h);

    //! draw call
    void display();

    //! release all rendering resources
    void dispose();

  public:
    float t;
    int selectedOutput;
    std::string parameters_path_;

  private:
    int windowWidth;
    int windowHeight;

    struct ShaderSettings {
      int width = 512;
      int height = 512;
      bool wireframe = false;
      Matrix backgroundColor; 
      std::string nodeClassName;
      std::string nodeName;
    };

    ShaderNode shaderNodeA;
    Mesh meshA;
    ShaderSettings shaderSettingsA;
   
    ShaderNode shaderNodeB;
    Mesh meshB;
    ShaderSettings shaderSettingsB;

    ShaderSettings loadShaderSettings(const std::string& filename) const;
  };
}
#endif
