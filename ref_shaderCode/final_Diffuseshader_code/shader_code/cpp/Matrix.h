// Copyright (C) Thorsten Thormaehlen, MIT License (see license file)

#ifndef DEF_MATRIX_H
#define DEF_MATRIX_H

#include <vector>
#include <string>

namespace gsn {
  /*!
  \class Matrix Matrix.h
  \brief This class defines a matrix

  This class defines a matrix. A matrix with only one column represents a vector.
  The elements are stored in a float vector in column major order, e.g.
  for a 4 x 4 matrix the vector indices are:

  | 0  4  8 12 |
  | 1  5  9 13 |
  | 2  6 10 14 |
  | 3  7 11 15 |

  */

  class Matrix {
  public:

    //! constructor
    Matrix(int rows = 4, int columns = 4);
    
    //! constructor for a 3-vector
    Matrix(float x, float y, float z);

    //! constructor for a 4-vector
    Matrix(float x, float y, float z, float w);

    //! destructor
    ~Matrix() {};

    //! resize the matrix
    void resize(int rows, int columns);

    //! fill the matrix with a given value
    void fill(float value);

    //! set the value of an element at the given row and column
    void set(int row, int column, float value);

    //! get the value of an element at the given row and column
    float get(int row, int column);

    //! set the matrix to the identity matrix
    void setIdentity();

    //! returns the transposed matrix
    Matrix transpose() const;

    //! returns the inverted matrix
    Matrix invert() const;

    //! matrix-matrix addition
    Matrix operator+(const Matrix& rhs) const;

    //! matrix-matrix subtraction
    Matrix operator-(const Matrix& rhs) const;

    //! matrix-matrix multiplication
    Matrix operator*(const Matrix& rhs) const;

    //! matrix-scalar multiplication
    Matrix operator*(float scalar) const;

    //! scalar-matrix multiplication
    friend Matrix operator*(float scalar, Matrix rhs);
    
    //! compute the dot product of the first column vector
    float dot(const Matrix& rhs) const;

    //! compute the dot product for each column vector
    Matrix columnDot(const Matrix& rhs) const;

    //! compute the cross product for each column vector
    Matrix cross(const Matrix& rhs) const;

    //! normalize the length of each column vector to 1
    void columnNormalize();

    //! generates a 4 x 4 perspective matrix
    void setPerspective(float fov, float aspect, float zNear, float zFar);

    //! generates a 4x4 lookat matrix
    void setLookAt(float eyeX, float eyeY, float eyeZ, float centerX, float centerY, float centerZ, float upX, float upY, float upZ);

    //! convert matrix to a string for output
    std::string prettyString();

  public:
    int rows = 0;
    int cols = 0;
    std::vector <float> e;
  };
}
#endif
