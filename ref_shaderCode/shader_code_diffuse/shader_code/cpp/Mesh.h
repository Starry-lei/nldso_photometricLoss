// Copyright (C) Thorsten Thormaehlen, MIT License (see license file)

#ifndef DEF_MESH_H
#define DEF_MESH_H

#include <stdio.h>
#include <string>
#include <vector>

namespace gsn {
  /*!
  \class Mesh Mesh.h
  \brief This class defines a 3D mesh.

  This class holds all information to describe a 3D mesh. The data structure is quite general and is to
  some extent inspired by how meshes are represented in COLLADA. The names of the entities are chosen
  such that they match OpenGL's lingo, e.g., there are ArrayBuffers, ElementArrayBuffers, and VertexBuffers.
  Typically, the vertex data is stored in ArrayBuffers (of type float) and vertex indices are stored in
  ElementArrayBuffers (of type unsigned int). A VertexBuffer defines the semantic meaning of an ArrayBuffer,
  e.g, whether the ArrayBuffer contains positions, normals, or texture corrdinates, etc. 
  COLLADA allows lines, polygons, polylist, triangles, etc. to be elements of a mesh. 
  This might be added in future but, currently, this data structure supports only triangles and quads 
  that are organized in triangle groups and quad groups.
  */

  class Mesh {
  public:

    //! constructor
    Mesh() {};

    //! destructor
    ~Mesh() {};

    //! clears all mesh data
    void clearAll();

    //! updates triangle data from quads
    void updateTrianglesFromQuads();

    //! updates the data for wireframe rendering
    void updateWireframeBuffers();

    //! updates the OpenGL buffers
    void updateOpenGLBuffers();

    //! updates the internal mesh data
    void updateAllInternalData();

    //! creates a new mesh from given vertex data
    void createFromArrays(std::string groupName, int numberOfPrimitives, const std::vector<float>& positions,
      const std::vector<float>& normals, const std::vector<float>& texCoords, const std::vector<unsigned int>& indices, bool triGroupFlag);

    //! creates a mesh containing a single squared quadrilateral
    void createQuad();

  public:
    struct ArrayBuffer {
      std::vector<float> arrayBuffer;
      unsigned int arrayBufferOpenGL = 0;
    };

    struct ElementArrayBuffer {
      std::vector<unsigned int> elementArrayBuffer;
      unsigned int elementArrayBufferOpenGL = 0;
    };

    struct VertexBuffer {
      std::string semantic; // the semantic meaning of the data, e.g., "positions", "normals", "texcoord", etc.
      int arrayBufferID; // reference to array buffer (vertex data)
      int stride; // stride of vertex data (integer number)
      int offset; // offset of vertex data (integer number)
      int elementSize; // size of one element, must be set to 3 for positions, 3 for normals, and 2 for texcoords (integer number)
    };

    struct TriangleGroup {
      std::string groupName; // name of group
      int elementArrayBufferID = -1; // reference to element array buffer (containing the vertex indices)
      int noOfTriangles; // number of triangles (integer number)
      int offset; // offset in element array buffer or array buffer (integer number)
      int wireframeElementArrayBufferID = -1; // reference to element array buffer for wireframe rendering 
    };

    struct QuadGroup {
      std::string groupName; // name of group
      int elementArrayBufferID; // reference to element array buffer (containing the vertex indices)
      int noOfQuads;  // number of quads (integer number)
      int offset;  // offset in element array buffer or array buffer (integer number)
      int triangleGroupID = -1; // reference to triangle group representing the same data  (-1 for automatic creation)
    };

    std::vector <ArrayBuffer> arrayBuffers;
    std::vector <ElementArrayBuffer> elementArrayBuffers;
    std::vector <VertexBuffer> vertexBuffers;
    std::vector <TriangleGroup> triangleGroups;
    std::vector <QuadGroup> quadGroups;
  };
}
#endif
