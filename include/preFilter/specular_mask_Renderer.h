// Copyright (C) Thorsten Thormaehlen, MIT License (see license file)

#ifndef DEF_SPECULAR_MASK_RENDERER_H
#define DEF_SPECULAR_SPECULARMASK_H

#include <stdio.h>
#include <string>
#include <vector>
#include "preFilter/Mesh.h"
#include "preFilter/ShaderNode.h"

namespace gsn {
  /*!
  \class Renderer Renderer.h
  \brief This class defines a renderer.
  */

  class specular_Mask_Renderer {
  public:

    //! constructor
    specular_Mask_Renderer();

    //! destructor
    ~specular_Mask_Renderer();

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
    float roughness;
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
