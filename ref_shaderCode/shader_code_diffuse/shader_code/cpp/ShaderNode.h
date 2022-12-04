// Copyright (C) Thorsten Thormaehlen, MIT License (see license file)

#ifndef DEF_SHADERNODE_H
#define DEF_SHADERNODE_H

#include <map>
#include <vector>
#include "Mesh.h"
#include "Matrix.h"

namespace gsn {
  /*!
  \class ShaderNode ShaderNode.h
  \brief This class defines a shader
  
  This class is a C++ version of the GSN shader plugin node 
  https://www.gsn-lib.org/docs/nodes/ShaderPluginNode.php
  The output is rendered to an OpenGL framebuffer object (FBO). The FBO render targets can be
  inputs to other shader nodes, which allows to create a connected graph of shaders.
  */

  class ShaderNode {
  public:

    //! constructor
    ShaderNode() {};

    //! destructor
    ~ShaderNode();

  public:
    //! initializes a shader node from the given shader source
    void setShaderSource(const std::string& vertexShader, const std::string& fragmentShader);

    //! initializes a shader node from the given shader source files
    void setShaderSourceFromFile(const std::string& vertexShaderFilename, const std::string& fragmentShaderFilename);

    //! sets an integer uniform variable
    void setUniformInteger(const std::string& name, int value);

    //! sets a float uniform variable
    void setUniformFloat(const std::string& name, float value);

    //! sets a boolean uniform variable
    void setUniformBool(const std::string& name, bool value);

    //! sets a image (sampler2D texture) uniform variable
    void setUniformImage(const std::string& name, unsigned int textureID, bool underExternalControl= true);

    //! sets a matrix uniform variable
    void setUniformMatrix(const std::string& name, const Matrix& value);

    //! sets uniform variables from a file
    void setUniformsFromFile(const std::string& filename);

    //! render call 
    void render(const Mesh& mesh, const Matrix &background, int width, int height, bool wireframe = false, bool renderToFBO = true);

    //! returns the texture of the FBO with the given index
    unsigned int getRenderTarget(int index) const;

    //! returns the texture of the FBO with the given name
    unsigned int getRenderTargetByName(const std::string& name) const;

  private:
    struct Texture {
      unsigned int texID = 0;
      int width = 0;
      int height = 0;
      int wrap_s;
      int wrap_t;
      int mag_filter;
      int min_filter;
      bool underExternalControl = true;
    };
    struct UniformVariable {
      int type;  // 0 = int, 1 = float, 2 = bool, 3 = Matrix, 4 = Texture
      int location;
      int intVal;
      float floatVal;
      bool boolVal;
      Matrix matVal;
      Texture texVal;
    };
    struct RenderOptions {
      bool depthtest = true;
      int srcFactor = -1;
      int dstFactor = -1;
      int rgbSrcFactor = -1;
      int rbgDstFactor = -1;
      int aSrcFactor = -1;
      int aDstFactor = -1;
      int cullFace = -1;
    };
    struct RenderTarget {
      std::string name;
      unsigned int texID = 0;
    };

    std::string vertexShader;
    std::string fragmentShader;
    unsigned int vertID = 0;
    unsigned int fragID = 0;
    unsigned int progID = 0;
    int positionAttr = -1;
    int normalAttr = -1;
    int texCoordAttr = -1;
    int colorAttr = -1;
    std::map<std::string, UniformVariable> uniforms;
    std::vector<RenderTarget> renderTargets;
    RenderOptions renderOptions;
    unsigned fbo = 0;
    int fboWidth = -1;
    int fboHeight = -1;

    void findAllUniforms(const std::string& code);
    void findTexturePara(UniformVariable& u, const std::string& line) const;
    void findRenderOptions(const std::string& code);
    void findAllRenderTargets(const std::string& code);

    void setupShaders();
    void findAllUniformLocations();
    void resizeFBO(int width, int height);
    void setAllUniforms();
    
  };
}
#endif
