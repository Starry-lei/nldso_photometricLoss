// Copyright (C) Thorsten Thormaehlen, MIT License (see license file)

#ifndef DEF_RENDERER_H
#define DEF_RENDERER_H

#include <stdio.h>
#include <string>
#include <vector>
#include "Mesh.h"
#include "ShaderNode.h"

namespace gsn {
  /*!
  \class Renderer Renderer.h
  \brief This class defines a renderer.
  */

  class Renderer {
  public:

    //! constructor
    Renderer();

    //! destructor
    ~Renderer();

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
