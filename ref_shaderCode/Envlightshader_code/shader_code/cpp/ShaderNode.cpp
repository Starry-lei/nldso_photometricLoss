// Copyright (C) Thorsten Thormaehlen, MIT License (see license file)

#include <string>
#include <iostream> 
#include <iomanip>
#include <fstream>
#include <sstream>
#include <tuple>
#include <map>
#include <algorithm>

using namespace std;

#include <GL/glew.h>

#include "ShaderNode.h"
#include "StringTools.h"
#include "FileTools.h"
#include "Mesh.h"
using namespace gsn;

// some static helper functions
namespace gsn {
  void printShaderInfoLog(GLuint obj) {
    int infoLogLength = 0;
    int returnLength = 0;
    char* infoLog;
    glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &infoLogLength);
    if (infoLogLength > 0) {
      infoLog = (char*)malloc(infoLogLength);
      glGetShaderInfoLog(obj, infoLogLength, &returnLength, infoLog);
      printf("%s\n", infoLog);
      free(infoLog);
      exit(1);
    }
  }

  void printProgramInfoLog(GLuint obj) {
    int infoLogLength = 0;
    int returnLength = 0;
    char* infoLog;
    glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &infoLogLength);
    if (infoLogLength > 0) {
      infoLog = (char*)malloc(infoLogLength);
      glGetProgramInfoLog(obj, infoLogLength, &returnLength, infoLog);
      printf("%s\n", infoLog);
      free(infoLog);
      exit(1);
    }
  }

  bool checkFramebufferStatus() {
    GLenum status;
    status = (GLenum)glCheckFramebufferStatus(GL_FRAMEBUFFER);
    switch (status) {
    case GL_FRAMEBUFFER_COMPLETE:
      return true;
    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
      printf("Framebuffer incomplete, incomplete attachment\n");
      return false;
    case GL_FRAMEBUFFER_UNSUPPORTED:
      printf("Unsupported framebuffer format\n");
      return false;
    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
      printf("Framebuffer incomplete, missing attachment\n");
      return false;
    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
      printf("Framebuffer incomplete, missing draw buffer\n");
      return false;
    case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
      printf("Framebuffer incomplete, missing read buffer\n");
      return false;
    }
    return false;
  }

  string findAttributeInComments(const std::string& line, const std::string& attr)
  {
    size_t index = StringTools::toLower(line).find(attr);
    if (index != std::string::npos) {
      string part = line.substr(index);
      size_t start = part.find("\"");
      if (start != std::string::npos) {
        string nextpart = part.substr(start + 1);
        size_t end = nextpart.find("\"");
        if (end != std::string::npos) {
          return nextpart.substr(0, end);
        }
      }

    }
    return "";
  }

  int strToBendFunc(const std::string& inputStr) {
    string str = StringTools::toLower(inputStr);
    int ret = GL_ONE;
    if (str.find("zero") != std::string::npos) {
      ret = GL_ZERO;
    }
    if (str.find("one") != std::string::npos) {
      ret = GL_ONE;
    }
    if (str.find("src_color") != std::string::npos && str.find("one_minus_src_") == std::string::npos) {
      ret = GL_SRC_COLOR;
    }
    if (str.find("one_minus_src_color") != std::string::npos) {
      ret = GL_ONE_MINUS_SRC_COLOR;
    }
    if (str.find("dst_color") != std::string::npos) {
      ret = GL_DST_COLOR;
    }
    if (str.find("one_minus_dst_color") != std::string::npos) {
      ret = GL_ONE_MINUS_DST_COLOR;
    }
    if (str.find("src_alpha") != std::string::npos) {
      ret = GL_SRC_ALPHA;
    }
    if (str.find("one_minus_src_alpha") != std::string::npos) {
      ret = GL_ONE_MINUS_SRC_ALPHA;
    }
    if (str.find("dst_alpha") != std::string::npos) {
      ret = GL_DST_ALPHA;
    }
    if (str.find("one_minus_dst_alpha") != std::string::npos) {
      ret = GL_ONE_MINUS_DST_ALPHA;
    }

    return ret;
  }

  bool beforeGLSLversion1_2(const std::string &code) {
    const std::vector<string>& lines = StringTools::split(code, "\n");
    for (int i = 0; i < int(lines.size()); i++) {
      string line = lines[i];
      const std::vector<string>& lineParts = StringTools::split(line);
      if (lineParts.size() > 0) {
        if (lineParts[0] == "attribute") {
          return true;
        }
        if (lineParts[0] == "varying") {
          return true;
        }
      }
    }
    return false;
  }

}


ShaderNode::~ShaderNode()
{
  if (progID > 0) {
    glDeleteProgram(progID);
  }
  if (vertID > 0) {
    glDeleteShader(vertID);
  }
  if (fragID > 0) {
    glDeleteShader(fragID);
  }

  for (auto& i : uniforms) {
    UniformVariable& u = i.second;
    if (u.texVal.texID > 0) {
      if (!u.texVal.underExternalControl) {
        glDeleteTextures(1, &u.texVal.texID);
      }
    }
  }

  for (int r = 0; r < int(renderTargets.size()); r++) {
    RenderTarget& renderTarget = renderTargets[r];
    if (renderTarget.texID > 0) {
      glDeleteTextures(1, &renderTarget.texID);
    }
  }

  if (fbo > 0) {
    glDeleteFramebuffers(1, &fbo);
  }
}

void ShaderNode::setShaderSource(const std::string& _vertexShader, const std::string& _fragmentShader)
{
  vertexShader = _vertexShader;
  fragmentShader = _fragmentShader;

  if (vertexShader.size() == 0) {
    // if vertex source is not provided, use default
    if (beforeGLSLversion1_2(fragmentShader)) {
      vertexShader = "attribute vec3 position;\n";
      vertexShader += "attribute vec2 texcoord;\n";
      vertexShader += "varying vec2 tc;\n";
      vertexShader += "void main()\n";
      vertexShader += "{\n";
      vertexShader += "  tc = texcoord;\n";
      vertexShader += "  gl_Position = vec4(position, 1.0);\n";
      vertexShader += "}\n";
    }
    else {
      vertexShader = "#version 300 es\n";
      vertexShader += "in vec3 position;\n";
      vertexShader += "in vec2 texcoord;\n";
      vertexShader += "out vec2 tc;\n";
      vertexShader += "void main()\n";
      vertexShader += "{\n";
      vertexShader += "  tc = texcoord;\n";
      vertexShader += "  gl_Position = vec4(position, 1.0);\n";
      vertexShader += "}\n";
    }
  }

  findAllUniforms(vertexShader);
  findAllUniforms(fragmentShader);
  findRenderOptions(vertexShader + "\n" + fragmentShader);
  findAllRenderTargets(fragmentShader);

  setupShaders();
  findAllUniformLocations();
}




void ShaderNode::setShaderSourceFromFile(const std::string& vertexShaderFilename, const std::string& fragmentShaderFilename) {
  string vertexShader = "";
  if (vertexShaderFilename.size() > 0) {
    vertexShader = FileTools::readTextFile(vertexShaderFilename);
  }
  string fragmentShader = FileTools::readTextFile(fragmentShaderFilename);
  setShaderSource(vertexShader, fragmentShader);
}

void ShaderNode::findTexturePara(UniformVariable& u,  const std::string& line) const{

  u.texVal.wrap_s = GL_CLAMP_TO_EDGE; // default
  string wraps = findAttributeInComments(line, "wrap_s");
  if (int(wraps.size()) > 0) {
    string str = StringTools::toLower(wraps);
    size_t pos = str.find("border");
    if (pos != std::string::npos) {
      u.texVal.wrap_s = GL_CLAMP_TO_BORDER;
    }
    pos = str.find("repeat");
    if (pos != std::string::npos) {
      u.texVal.wrap_s = GL_REPEAT;
    }
    pos = str.find("mirror");
    if (pos != std::string::npos) {
      u.texVal.wrap_s = GL_MIRRORED_REPEAT;
    }
  }

  u.texVal.wrap_t = GL_CLAMP_TO_EDGE; // default
  string wrapt = findAttributeInComments(line, "wrap_t");
  if (int(wrapt.size()) > 0) {
    string str = StringTools::toLower(wrapt);
    size_t pos = str.find("border");
    if (pos != std::string::npos) {
      u.texVal.wrap_t = GL_CLAMP_TO_BORDER;
    }
    pos = str.find("repeat");
    if (pos != std::string::npos) {
      u.texVal.wrap_t = GL_REPEAT;
    }
    pos = str.find("mirror");
    if (pos != std::string::npos) {
      u.texVal.wrap_t = GL_MIRRORED_REPEAT;
    }
  }

  u.texVal.mag_filter = GL_NEAREST;
  string magfilter = findAttributeInComments(line, "mag_filter");
  if (int(magfilter.size()) > 0) {
    string str = StringTools::toLower(magfilter);
    size_t pos = str.find("linear");
    if (pos != std::string::npos) {
      u.texVal.mag_filter = GL_LINEAR;
    }
  }

  u.texVal.min_filter = GL_LINEAR_MIPMAP_NEAREST;
  string minfilter = findAttributeInComments(line, "min_filter");
  if (int(minfilter.size()) > 0) {
    string str = StringTools::toLower(minfilter);
    size_t posnear = str.find("nearest");
    size_t posmip = str.find("mipmap");
    size_t poslinmip = str.find("linear_mipmap_linear");
    size_t poslin = str.find("linear");

    if (posnear != std::string::npos && posmip == std::string::npos) {
      u.texVal.min_filter =  GL_NEAREST;
    }
    if (poslin != std::string::npos && posmip == std::string::npos) {
      u.texVal.min_filter = GL_LINEAR;
    }
    if (poslinmip != std::string::npos) {
      u.texVal.min_filter = GL_LINEAR_MIPMAP_LINEAR;
    }
  }
}

void ShaderNode::findAllUniforms(const std::string& str) {
  const std::vector<string>& lines = StringTools::split(str, "\n");
  for (int i = 0; i < int(lines.size()); i++) {
    string line = lines[i];
    const std::vector<string>& lineParts = StringTools::split(line);
    if (int(lineParts.size()) >= 3) {
      if (lineParts[0] == "uniform") {
        string type = lineParts[1];
        string name = lineParts[2];
        std::size_t nameEndIndex = name.find(";");
        if (nameEndIndex != std::string::npos) {
          name = name.substr(0, nameEndIndex);
        }

        UniformVariable u;
        u.location = -1;
        string attr = findAttributeInComments(line, "defaultval");
        // 0 = int, 1 = float, 2 = bool, 3 = Matrix, 4 = Image
        if (type == "int") {
          u.type = 0;
          if (attr.length() > 0) {
            u.intVal = StringTools::stringToInt(attr);
          }
        }
        if (type == "float") {
          u.type = 1;
          if (attr.length() > 0) {
            u.floatVal = StringTools::stringToFloat(attr);
          }
        }
        if (type == "bool") {
          u.type = 2;
          if (attr.length() > 0) {
            u.boolVal = StringTools::stringToBool(attr);
          }
        }
        if (type == "vec2") {
          u.type = 3;
          u.matVal.resize(2, 1);
        }
        if (type == "vec3") {
          u.type = 3;
          u.matVal.resize(3, 1);
        }
        if (type == "vec4") {
          u.type = 3;
          u.matVal.resize(4, 1);
        }
        if (type == "mat4") {
          u.type = 3;
          u.matVal.resize(4, 4);
        }
        if (type == "sampler2D") {
          u.type = 4;
          findTexturePara(u, line);
        }

        if (u.type == 3 && attr.length() > 0) {
          const std::vector<string>& elements = StringTools::split(attr, ",");
          int num = std::min(int(elements.size()), int(u.matVal.e.size()));
          for (int n = 0; n < num; n++) {
            u.matVal.e[n] = StringTools::stringToFloat(elements[n]);
          }
        }
        uniforms.insert(std::make_pair(name, u));
      }
    }
  }
}

void ShaderNode::findRenderOptions(const std::string& code)
{
  // find all render options variables
  renderOptions.depthtest = true;
  renderOptions.srcFactor = GL_ONE;
  renderOptions.dstFactor = GL_ONE_MINUS_SRC_ALPHA;
  renderOptions.rgbSrcFactor = -1;
  renderOptions.rbgDstFactor = -1;
  renderOptions.aSrcFactor = -1;
  renderOptions.aDstFactor = -1;
  renderOptions.cullFace = -1;
  const std::vector<string>& lines = StringTools::split(code, "\n");
  for (int i = 0; i < int(lines.size()); i++) {
    string line = lines[i];
    size_t index = StringTools::toLower(line).find("gsnshaderoption"); // also finds "gsnShaderOptions" with "s"
    if (index != std::string::npos) {
      string attr = StringTools::toLower(findAttributeInComments(line, "depth_test"));
      if (attr.find("false") != std::string::npos || attr.find("disable") != std::string::npos) {
        renderOptions.depthtest = false;
      }
      attr = findAttributeInComments(line, "blend_func");
      if (attr.length() > 0) {
        const std::vector<string>& elements = StringTools::split(attr, ",");
        if (int(elements.size()) >= 2) {
          renderOptions.srcFactor = strToBendFunc(elements[0]);
          renderOptions.dstFactor = strToBendFunc(elements[1]);
          renderOptions.rgbSrcFactor = -1;
          renderOptions.rbgDstFactor = -1;
          renderOptions.aSrcFactor = -1;
          renderOptions.aDstFactor = -1;
        }
      }
      attr = findAttributeInComments(line, "blend_func_separate");
      if (attr.length() > 0) {
        const std::vector<string>& elements = StringTools::split(attr, ",");
        if (int(elements.size()) >= 4) {
          renderOptions.srcFactor = -1;
          renderOptions.dstFactor = -1;
          renderOptions.rgbSrcFactor = strToBendFunc(elements[0]);
          renderOptions.rbgDstFactor = strToBendFunc(elements[1]);
          renderOptions.aSrcFactor = strToBendFunc(elements[2]);
          renderOptions.aDstFactor = strToBendFunc(elements[3]);
        }
      }
      attr = StringTools::toLower(findAttributeInComments(line, "cull_face"));
      if (attr.length() > 0) {
        if (attr.find("back") != std::string::npos) {
          renderOptions.cullFace = GL_BACK;
        }
        if (attr.find("front") != std::string::npos) {
          renderOptions.cullFace = GL_FRONT;
        }
        if (attr.find("front_and_back") != std::string::npos) {
          renderOptions.cullFace = GL_FRONT_AND_BACK;
        }
      }
    }
  }
}

void ShaderNode::findAllRenderTargets(const std::string& code)
{
  for (int r = 0; r < int(renderTargets.size()); r++) {
    RenderTarget& renderTarget = renderTargets[r];
    if (renderTarget.texID > 0) {
      glDeleteTextures(1, &renderTarget.texID);
    }
  }
  renderTargets.clear();

  const std::vector<string>& lines = StringTools::split(code, "\n");
  for (int i = 0; i < int(lines.size()); i++) {
    string line = lines[i];
    const std::vector<string>& lineParts = StringTools::split(line);
    for (int p = 0; p < int(lineParts.size()); p++) {
      if (lineParts[p] == "out") {
        if (int(lineParts.size()) > p + 2) {
          if (lineParts[p + 1] == "vec4") {
            string name = lineParts[p + 2];
            name = StringTools::trimRight(name, ";");
            RenderTarget renderTarget;
            renderTarget.name = name;
            renderTarget.texID = 0;
            renderTargets.push_back(renderTarget);
          }
        }
      }
    }
  }

  if (renderTargets.size() == 0) {
    RenderTarget renderTarget;
    renderTarget.name = "default";
    renderTarget.texID = 0;
    renderTargets.push_back(renderTarget);
  }

  RenderTarget renderTargetDepth;
  renderTargetDepth.name = "depth";
  renderTargetDepth.texID = 0;
  renderTargets.push_back(renderTargetDepth);

}

void ShaderNode::setupShaders() {

  if (vertID > 0) {
    glDeleteShader(vertID);
  }
  if (fragID > 0) {
    glDeleteShader(fragID);
  }

  // create shader
  vertID = glCreateShader(GL_VERTEX_SHADER);
  fragID = glCreateShader(GL_FRAGMENT_SHADER);

  // specify shader source
  const char* vss = vertexShader.c_str();
  const char* fss = fragmentShader.c_str();
  glShaderSource(vertID, 1, &(vss), NULL);
  glShaderSource(fragID, 1, &(fss), NULL);

  // compile the shader
  glCompileShader(vertID);
  glCompileShader(fragID);

  // check for errors
  printShaderInfoLog(vertID);
  printShaderInfoLog(fragID);


  if (progID > 0) {
    glDeleteProgram(progID);
  }

  // create program and attach shaders
  progID = glCreateProgram();
  glAttachShader(progID, vertID);
  glAttachShader(progID, fragID);

  // link the program
  glLinkProgram(progID);
  // output error messages
  printProgramInfoLog(progID);

  positionAttr = glGetAttribLocation(progID, "position");
  normalAttr = glGetAttribLocation(progID, "normal");
  texCoordAttr = glGetAttribLocation(progID, "texcoord");
  colorAttr = glGetAttribLocation(progID, "color");
}

void ShaderNode::resizeFBO(int width, int height) {

  if (width == fboWidth && height == fboHeight) {
    return;
  }
  else {
    fboWidth = width;
    fboHeight = height;
  }

  for (int r = 0; r < int(renderTargets.size()); r++) {
    RenderTarget& renderTarget = renderTargets[r];
    if (renderTarget.texID > 0) {
      glDeleteTextures(1, &renderTarget.texID);
      renderTarget.texID = 0;
    }
  }

  if (fbo > 0) {
    glDeleteFramebuffers(1, &fbo);
  }

  // create a frame buffer object
  glGenFramebuffers(1, &fbo);

  // bind the frame buffer
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);

  int colorBufferCounter = 0;
  for (int r = 0; r < int(renderTargets.size()); r++) {
    RenderTarget& renderTarget = renderTargets[r];

    if (renderTarget.name != "depth") {
      GLuint texID;
      glGenTextures(1, &texID);
      glBindTexture(GL_TEXTURE_2D, texID);

      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, fboWidth, fboHeight, 0, GL_RGBA, GL_FLOAT, NULL);
      // Attach the texture to the fbo
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + colorBufferCounter, GL_TEXTURE_2D, texID, 0);
      if (!checkFramebufferStatus()) exit(1);
      renderTarget.texID = texID;
      colorBufferCounter++;
    } else {
      GLuint depthID;
      // Generate depth render texture
      glGenTextures(1, &depthID);
      glBindTexture(GL_TEXTURE_2D, depthID);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, fboWidth, fboHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

      // Attach the texture to the fbo
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthID, 0);
      if (!checkFramebufferStatus()) exit(1);
      renderTarget.texID = depthID;
    }
  }
  // unbind texture
  glBindTexture(GL_TEXTURE_2D, 0);
  //unbind fbo
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ShaderNode::findAllUniformLocations()
{
  for (auto& i : uniforms) {
    UniformVariable& u = i.second;
    u.location = glGetUniformLocation(progID, i.first.c_str());
    if (u.location < 0) {
      cout << "ShaderNode::findAllUniformLocations: Unresolved location of uniform \"" + i.first  << "\". The GLSL compiler might have removed unused uniforms." << endl;
    }
  }
}

void ShaderNode::setAllUniforms()
{
  int texCount = 0;
  for (auto& i : uniforms) {
    UniformVariable& u = i.second;
    if (u.location != -1) {
      if (u.type == 0) { // "int"
        glUniform1i(u.location, u.intVal);
      }
      if (u.type == 1) { // "float"
        glUniform1f(u.location, u.floatVal);
      }
      if (u.type == 2) { // "bool"
        if (u.boolVal) {
          glUniform1i(u.location, 1);
        }
        else {
          glUniform1i(u.location, 0);
        }
      }
      if (u.type == 3) {
        if (u.matVal.rows == 2 && u.matVal.cols == 1) { // "vec2"
          glUniform2fv(u.location, 1, &u.matVal.e[0]);
        }
        if (u.matVal.rows == 3 && u.matVal.cols == 1) { // "vec3"
          glUniform3fv(u.location, 1, &u.matVal.e[0]);
        }
        if (u.matVal.rows == 4 && u.matVal.cols == 1) { // "vec4"
          glUniform4fv(u.location, 1, &u.matVal.e[0]);
        }
        if (u.matVal.rows == 4 && u.matVal.cols == 4) { // "mat4"
          glUniformMatrix4fv(u.location, 1, false, &u.matVal.e[0]);
        }
      }
      if (u.type == 4) {
        if (u.texVal.texID > 0) {
          glActiveTexture(GL_TEXTURE0 + texCount);
          // set texture parameters
          glBindTexture(GL_TEXTURE_2D, u.texVal.texID);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, u.texVal.wrap_s);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, u.texVal.wrap_t);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, u.texVal.mag_filter);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, u.texVal.min_filter);
          glUniform1i(u.location, texCount);
          texCount++;
        }
      }
    }
  }
}


void ShaderNode::render(const Mesh& mesh, const Matrix& background, int width, int height, bool wireframe, bool renderToFBO)
{
  setUniformInteger("width", width);
  setUniformInteger("height", height);

  if (renderToFBO) {
    resizeFBO(width, height);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    std::vector<unsigned int> renderTar(renderTargets.size() - 1);
    for (int r = 0; r < int(renderTargets.size() - 1); r++) {
      renderTar[r] = GL_COLOR_ATTACHMENT0 + r;
    }
    // set the color attachments to write to
    glDrawBuffers(int(renderTargets.size()) - 1, &renderTar[0]);
  }
  else {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }
  if (background.rows == 4 && background.cols == 1) {
    glClearColor(background.e[0], background.e[1], background.e[2], background.e[3]);
  }

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glEnable(GL_TEXTURE_2D);

  if (renderOptions.depthtest) {
    glEnable(GL_DEPTH_TEST);
  } else {
    glDisable(GL_DEPTH_TEST);
  }

  glEnable(GL_BLEND);
  if (renderOptions.srcFactor >= 0) {
    glBlendFunc(renderOptions.srcFactor, renderOptions.dstFactor);
  }
  if (renderOptions.rgbSrcFactor >= 0) {
    glBlendFuncSeparate(renderOptions.rgbSrcFactor, renderOptions.rbgDstFactor, renderOptions.aSrcFactor, renderOptions.aDstFactor);
  }
  if (renderOptions.cullFace >= 0) {
    glEnable(GL_CULL_FACE);
    glCullFace(renderOptions.cullFace);
  }

  glViewport(0, 0, width, height);

  glUseProgram(progID);

  std::vector <int> enabledAttribArray;
  for (int i = 0; i < int(mesh.vertexBuffers.size()); i++) {
    const Mesh::VertexBuffer& vb = mesh.vertexBuffers[i];
    int loc = -1;
    if (vb.semantic == "position")
      loc = positionAttr;
    if (vb.semantic == "normal")
      loc = normalAttr;
    if (vb.semantic == "texcoord")
      loc = texCoordAttr;
    if (vb.semantic == "color")
      loc = colorAttr;
    if (loc >= 0) {
      glBindBuffer(GL_ARRAY_BUFFER, mesh.arrayBuffers[vb.arrayBufferID].arrayBufferOpenGL);
      glVertexAttribPointer(loc, vb.elementSize, GL_FLOAT, false, vb.stride * sizeof(float), (GLvoid *) (vb.offset * sizeof(float)));
      glEnableVertexAttribArray(loc);
      enabledAttribArray.push_back(loc);
    }
  }

  
  for (int i = 0; i < int(mesh.triangleGroups.size()); i++) {

    setUniformInteger("gsnMeshGroup", i);
    setAllUniforms();

    const Mesh::TriangleGroup& tg = mesh.triangleGroups[i];
    if (!wireframe) {
      if (tg.elementArrayBufferID < 0) {
        glDrawArrays(GL_TRIANGLES, tg.offset * 3, tg.noOfTriangles * 3);
      }
      else {
        const Mesh::ElementArrayBuffer& eb = mesh.elementArrayBuffers[tg.elementArrayBufferID];
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eb.elementArrayBufferOpenGL);
        glDrawElements(GL_TRIANGLES, tg.noOfTriangles * 3, GL_UNSIGNED_INT, NULL /*(GLvoid*) (tg.offset * 3)*/);
      }
    }
    else {
      const Mesh::ElementArrayBuffer& eb = mesh.elementArrayBuffers[tg.wireframeElementArrayBufferID];
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eb.elementArrayBufferOpenGL);
      glDrawElements(GL_LINES, int(eb.elementArrayBuffer.size()), GL_UNSIGNED_INT, 0);
    }
  }

  for (int i = 0; i < int(enabledAttribArray.size()); i++) {
    glDisableVertexAttribArray(enabledAttribArray[i]);
  }

  if (renderToFBO) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }

  for (int r = 0; r < int(renderTargets.size()); r++) {
    RenderTarget& renderTarget = renderTargets[r];
    if (renderTarget.texID > 0) {
      glBindTexture(GL_TEXTURE_2D, renderTarget.texID);
      glGenerateMipmap(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, 0);
    }
  }

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glDisable(GL_CULL_FACE);
}

void ShaderNode::setUniformInteger(const std::string& name, int value) 
{
  if (uniforms.count(name) > 0) {
    if (uniforms.at(name).type == 0) {
      uniforms.at(name).intVal = value;
    }
  }
}

void ShaderNode::setUniformFloat(const std::string& name, float value)
{
  if (uniforms.count(name) > 0) {
    if (uniforms.at(name).type == 1) {
      uniforms.at(name).floatVal = value;
    }
  }
}

void ShaderNode::setUniformBool(const std::string& name, bool value)
{
  if (uniforms.count(name) > 0) {
    if (uniforms.at(name).type == 2) {
      uniforms.at(name).boolVal = value;
    }
  }
}

void ShaderNode::setUniformMatrix(const std::string& name, const Matrix& value)
{
  if (uniforms.count(name) > 0) {
    if (uniforms.at(name).type == 3) {
      uniforms.at(name).matVal = value;
    }
  }
  if (uniforms.count(name + "Inverse") > 0) {
    if (uniforms.at(name + "Inverse").type == 3) {
      uniforms.at(name + "Inverse").matVal = value.invert();
    }
  }
  if (uniforms.count(name + "TransposedInverse") > 0) {
    if (uniforms.at(name + "TransposedInverse").type == 3) {
      uniforms.at(name + "TransposedInverse").matVal = value.transpose().invert();
    }
  }
  if (uniforms.count(name + "InverseTransposed") > 0) {
    if (uniforms.at(name + "InverseTransposed").type == 3) {
      uniforms.at(name + "InverseTransposed").matVal = value.invert().transpose();
    }
  }
}

void ShaderNode::setUniformImage(const std::string& name, unsigned int textureID, bool underExternalControl) {
  if (uniforms.count(name) > 0) {
    UniformVariable& u = uniforms.at(name);
    if (u.type == 4) {
      u.texVal.texID = textureID;
      u.texVal.underExternalControl = underExternalControl;
    }
  }
  if (uniforms.count(name + "Width") > 0) {
    UniformVariable& u = uniforms.at(name + "Width");
    if (u.type == 0) {
      int w;
      glBindTexture(GL_TEXTURE_2D, textureID);
      glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
      glBindTexture(GL_TEXTURE_2D, 0);
      u.intVal = w;
    }
  }
  if (uniforms.count(name + "Height") > 0) {
    UniformVariable& u = uniforms.at(name + "Height");
    if (u.type == 0) {
      int h;
      glBindTexture(GL_TEXTURE_2D, textureID);
      glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h);
      glBindTexture(GL_TEXTURE_2D, 0);
      u.intVal = h;
    }
  }
}

void ShaderNode::setUniformsFromFile(const std::string& filename) {
  std::string content = FileTools::readTextFile(filename);
  const std::vector <string> lines = StringTools::split(content, "\n");
  for (int i = 0; i < int(lines.size()); i++) {
    const std::string& line = lines[i];
    const std::vector <string> items = StringTools::split(line, ",");
    if (items.size() >= 2) {
      string name = items[0];
      string type = StringTools::toLower(items[1]);
      if (type == "integer" && items.size() >= 3) {
        int val = StringTools::stringToInt(items[2]);
        setUniformInteger(name, val);
      }
      if (type == "float" && items.size() >= 3) {
        float val = StringTools::stringToFloat(items[2]);
        setUniformFloat(name, val);
      }
      if (type == "boolean" && items.size() >= 3) {
        bool val = StringTools::stringToBool(items[2]);
         setUniformBool(name, val);
      }
      if (type == "color" && items.size() >= 6) {
        float x = StringTools::stringToFloat(items[2]);
        float y = StringTools::stringToFloat(items[3]);
        float z = StringTools::stringToFloat(items[4]);
        float w = StringTools::stringToFloat(items[5]);
        Matrix mat(x, y, z, w);
        setUniformMatrix(name, mat);
      }
      if (type == "matrix" && items.size() >= 4) {
        int rows = StringTools::stringToInt(items[2]);
        int cols = StringTools::stringToInt(items[3]);
        Matrix mat(rows, cols);
        if (int(items.size()) >= 4 + (rows * cols)) {
          for (int j = 0; j < int(mat.e.size()); j++) {
            mat.e[j] = StringTools::stringToFloat(items[4 + j]);
          }
          setUniformMatrix(name, mat);
        }
      }
      if (type == "image") {
        string imgFileName = FileTools::getDirectory(filename) + "/" + name + ".pfm";
        std::vector<float> data;
        int width = 0;
        int height = 0;
        FileTools::loadAlphaPFM(imgFileName, width, height, data);

        //request textureID
        GLuint textureID;
        glGenTextures(1, &textureID);

        // bind texture
        glBindTexture(GL_TEXTURE_2D, textureID);

        // specify the 2D texture map
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, &data[0]);
        glGenerateMipmap(GL_TEXTURE_2D);

        glBindTexture(GL_TEXTURE_2D, 0);

        setUniformImage(name, textureID, false);

      }
    }
  }
}

unsigned ShaderNode::getRenderTarget(int index) const {
  int i = index % int(renderTargets.size());
  return renderTargets[i].texID;
}

unsigned ShaderNode::getRenderTargetByName(const std::string& name) const {
  for (int i = 0; i < int(renderTargets.size()); i++) {
    if (renderTargets[i].name == name) {
      return renderTargets[i].texID;
    }
  }

  return -1;
}
