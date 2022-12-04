// Copyright (C) Thorsten Thormaehlen, MIT License (see license file)

#include <string>
#include <iostream> 
#include <sstream> 
#include <vector>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>

#include "Matrix.h"

using namespace std;
using namespace gsn;

Matrix::Matrix(int _rows, int _columns)
{
  rows = _rows;
  cols = _columns;
  e.resize(rows * cols);
}

Matrix::Matrix(float x, float y, float z) {
  rows = 3;
  cols = 1;
  e.resize(rows * cols);
  e[0] = x;
  e[1] = y;
  e[2] = z;
}

Matrix::Matrix(float x, float y, float z, float w) {
  rows = 4;
  cols = 1;
  e.resize(rows * cols);
  e[0] = x;
  e[1] = y;
  e[2] = z;
  e[3] = w;
}

void Matrix::resize(int _rows, int _columns) {
  rows = _rows;
  cols = _columns;
  e.resize(rows * cols);
}

void Matrix::fill(float value) {
  for (int i = 0; i < int(e.size()); i++) {
    e[i] = value;
  }
}

void Matrix::set(int row, int column, float value) {
  e[row + rows * column] = value;
}

float Matrix::get(int row, int column) {
  return e[row + rows * column];
}

void Matrix::setIdentity() {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      if (r == c) {
        e[r + rows * c] = 1.0;
      }
      else {
        e[r + rows * c] = 0.0;
      }
    }
  }
}

Matrix Matrix::transpose() const {
  Matrix result(cols, rows);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      result.e[c + result.rows * r] = e[r + rows * c];
    }
  }
  return result;
}

Matrix Matrix::operator*(const Matrix& mat) const {
  Matrix result(rows, mat.cols);
  if (cols == mat.rows) {
    float sum = 0.0;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < mat.cols; j++) {
        sum = 0.0;
        for (int k = 0; k < cols; k++) {
          sum += e[i + rows * k] * mat.e[k + mat.rows * j];
        }
        result.e[i + result.rows * j] = sum;
      }
    }
  }
  else {
    result.setIdentity();
  }
  return result;
}

Matrix Matrix::operator+(const Matrix& mat) const {
  int r = min(rows, mat.rows);
  int c = min(cols, mat.cols);
  Matrix result(r, c);
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      result.e[i + result.rows * j] = e[i + rows * j] + mat.e[i + mat.rows * j];
    }
  }
  return result;
}

Matrix Matrix::operator-(const Matrix& mat) const {
  int r = min(rows, mat.rows);
  int c = min(cols, mat.cols);
  Matrix result(r, c);
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      result.e[i + result.rows * j] = e[i + rows * j] - mat.e[i + mat.rows * j];
    }
  }
  return result;
}

Matrix Matrix::operator*(float scalar) const {
  Matrix result = *this;
  for (int r = 0; r < result.rows; r++) {
    for (int c = 0; c < result.cols; c++) {
      result.e[r + result.rows * c] *= scalar;
    }
  }
  return result;
}

namespace gsn {
  Matrix operator*(float scalar, Matrix rhs) {
    Matrix result = rhs;
    for (int r = 0; r < result.rows; r++) {
      for (int c = 0; c < result.cols; c++) {
        result.e[r + result.rows * c] *= scalar;
      }
    }
    return result;
  }
}

float Matrix::dot(const Matrix& mat) const {
  int r = min(rows, mat.rows);
  int c = min(cols, mat.cols);
  float result = 0.0;
  if (c > 0) {
    for (int i = 0; i < r; i++) {
      result += e[i] * mat.e[i];
    }
  }
  return result;
}

Matrix Matrix::columnDot(const Matrix& mat) const {
  int r = min(rows, mat.rows);
  int c = min(cols, mat.cols);
  Matrix result(1, c);

  for (int j = 0; j < c; j++) {
    result.e[0 + result.rows * j] = 0.0;
    for (int i = 0; i < r; i++) {
      result.e[0 + result.rows * j] += e[i + rows * j] * mat.e[i + mat.rows * j];
    }
  }
  return result;
}

Matrix Matrix::cross(const Matrix& mat) const {
  int r = min(rows, mat.rows);
  int c = min(cols, mat.cols);
  Matrix result(3, c);
  result.fill(0.0);
  if (r < 3) return result;

  for (int j = 0; j < c; j++) {
    result.e[0 + result.rows * j] = e[1 + rows * j] * mat.e[2 + mat.rows * j] - mat.e[1 + mat.rows * j] * e[2 + rows * j];
    result.e[1 + result.rows * j] = e[2 + rows * j] * mat.e[0 + mat.rows * j] - mat.e[2 + mat.rows * j] * e[0 + rows * j];
    result.e[2 + result.rows * j] = e[0 + rows * j] * mat.e[1 + mat.rows * j] - mat.e[0 + mat.rows * j] * e[1 + rows * j];
  }
  return result;
}

void Matrix::columnNormalize() {
  for (int c = 0; c < cols; c++) {
    float length = 0.0;
    for (int r = 0; r < rows; r++) {
      length += (e[r + rows * c] * e[r + rows * c]);
    }
    length = sqrt(length);
    if (abs(length) > 1e-32) {
      for (int r = 0; r < rows; r++) {
        e[r + rows * c] /= length;
      }
    }
  }
}

void Matrix::setPerspective(float fov, float aspect, float zNear, float zFar) {
  resize(4, 4);
  setIdentity();
  float f = 1.0f / (tan(fov / 2.0f * (float(M_PI) / 180.0f)));
  e[0] = f / aspect;
  e[5] = f;
  e[10] = (zFar + zNear) / (zNear - zFar);
  e[14] = (2.0f * zFar * zNear) / (zNear - zFar);
  e[11] = -1.0f;
  e[15] = 0.0f;
}

void Matrix::setLookAt(float eyeX, float eyeY, float eyeZ, float centerX, float centerY, float centerZ, float upX, float upY, float upZ) {
  
  Matrix up(upX, upY, upZ);
  Matrix eye(eyeX, eyeY, eyeZ);
  Matrix dir(centerX - eyeX, centerY - eyeY, centerZ - eyeZ);

  dir.columnNormalize();
  Matrix right = dir.cross(up);
  right.columnNormalize();
  up = right.cross(dir);
  up.columnNormalize();

  resize(4, 4);
  // first row
  e[0] = right.e[0];
  e[4] = right.e[1];
  e[8] = right.e[2];
  e[12] = -(right.dot(eye));
  // second row
  e[1] = up.e[0];
  e[5] = up.e[1];
  e[9] = up.e[2];
  e[13] = -(up.dot(eye));
  // third row
  e[2] = -dir.e[0];
  e[6] = -dir.e[1];
  e[10] = -dir.e[2];
  e[14] = dir.dot(eye);
  // forth row
  e[3] = 0.0;
  e[7] = 0.0;
  e[11] = 0.0;
  e[15] = 1.0;
}

std::string Matrix::prettyString() {
  std::stringstream buf;
  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < cols; y++) {
      buf << e[x + rows * y];
      if (y < cols - 1) {
        buf << " ";
      }
    }
    buf << std::endl;
  }
  return buf.str();
}

Matrix Matrix::invert() const
{
  Matrix result = Matrix(rows, cols);

  if (rows != cols) {
    // matrix is not squared
    cerr << "Matrix is not squared" << endl;
    return result;
  }

  std::vector<float>& inv = result.e;

  if (rows == 4 && cols == 4) {
    // because 4x4 is used most, here is a special implementation for it
    inv[0] = e[5] * e[10] * e[15] - e[5] * e[11] * e[14] - e[9] * e[6] * e[15] +
      e[9] * e[7] * e[14] + e[13] * e[6] * e[11] - e[13] * e[7] * e[10];
    inv[4] = -e[4] * e[10] * e[15] + e[4] * e[11] * e[14] + e[8] * e[6] * e[15] -
      e[8] * e[7] * e[14] - e[12] * e[6] * e[11] + e[12] * e[7] * e[10];
    inv[8] = e[4] * e[9] * e[15] - e[4] * e[11] * e[13] - e[8] * e[5] * e[15] +
      e[8] * e[7] * e[13] + e[12] * e[5] * e[11] - e[12] * e[7] * e[9];
    inv[12] = -e[4] * e[9] * e[14] + e[4] * e[10] * e[13] + e[8] * e[5] * e[14] -
      e[8] * e[6] * e[13] - e[12] * e[5] * e[10] + e[12] * e[6] * e[9];
    inv[1] = -e[1] * e[10] * e[15] + e[1] * e[11] * e[14] + e[9] * e[2] * e[15] -
      e[9] * e[3] * e[14] - e[13] * e[2] * e[11] + e[13] * e[3] * e[10];
    inv[5] = e[0] * e[10] * e[15] - e[0] * e[11] * e[14] - e[8] * e[2] * e[15] +
      e[8] * e[3] * e[14] + e[12] * e[2] * e[11] - e[12] * e[3] * e[10];
    inv[9] = -e[0] * e[9] * e[15] + e[0] * e[11] * e[13] + e[8] * e[1] * e[15] -
      e[8] * e[3] * e[13] - e[12] * e[1] * e[11] + e[12] * e[3] * e[9];
    inv[13] = e[0] * e[9] * e[14] - e[0] * e[10] * e[13] - e[8] * e[1] * e[14] +
      e[8] * e[2] * e[13] + e[12] * e[1] * e[10] - e[12] * e[2] * e[9];
    inv[2] = e[1] * e[6] * e[15] - e[1] * e[7] * e[14] - e[5] * e[2] * e[15] +
      e[5] * e[3] * e[14] + e[13] * e[2] * e[7] - e[13] * e[3] * e[6];
    inv[6] = -e[0] * e[6] * e[15] + e[0] * e[7] * e[14] + e[4] * e[2] * e[15] -
      e[4] * e[3] * e[14] - e[12] * e[2] * e[7] + e[12] * e[3] * e[6];
    inv[10] = e[0] * e[5] * e[15] - e[0] * e[7] * e[13] - e[4] * e[1] * e[15] +
      e[4] * e[3] * e[13] + e[12] * e[1] * e[7] - e[12] * e[3] * e[5];
    inv[14] = -e[0] * e[5] * e[14] + e[0] * e[6] * e[13] + e[4] * e[1] * e[14] -
      e[4] * e[2] * e[13] - e[12] * e[1] * e[6] + e[12] * e[2] * e[5];
    inv[3] = -e[1] * e[6] * e[11] + e[1] * e[7] * e[10] + e[5] * e[2] * e[11] -
      e[5] * e[3] * e[10] - e[9] * e[2] * e[7] + e[9] * e[3] * e[6];
    inv[7] = e[0] * e[6] * e[11] - e[0] * e[7] * e[10] - e[4] * e[2] * e[11] +
      e[4] * e[3] * e[10] + e[8] * e[2] * e[7] - e[8] * e[3] * e[6];
    inv[11] = -e[0] * e[5] * e[11] + e[0] * e[7] * e[9] + e[4] * e[1] * e[11] -
      e[4] * e[3] * e[9] - e[8] * e[1] * e[7] + e[8] * e[3] * e[5];
    inv[15] = e[0] * e[5] * e[10] - e[0] * e[6] * e[9] - e[4] * e[1] * e[10] +
      e[4] * e[2] * e[9] + e[8] * e[1] * e[6] - e[8] * e[2] * e[5];

    float det = e[0] * inv[0] + e[1] * inv[4] + e[2] * inv[8] + e[3] * inv[12];
    if (abs(det) < 1e-16) {
      cerr << "Singular Matrix" << endl;
      return result;
    }
    det = 1.0f / det;
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        inv[r + rows * c] *= det;
      }
    }
  }
  else {
    // inverse for general matrix sizes
    result = *this;

    int n = rows;
    std::vector<float> b(n);
    std::vector<int> indexCol(n);
    std::vector<int> indexRow(n);
    std::vector<int> pivot(n);
    int mRow = 0;
    int mCol = 0;
    float tmp = 0.0;
    for (int j = 0; j < n; j++) {
      pivot[j] = 0;
    }
    for (int i = 0; i < n; i++) {
      float bbb = 0.0;
      for (int j = 0; j < n; j++) {
        if (pivot[j] != 1) {
          for (int k = 0; k < n; k++) {
            if (pivot[k] == 0) {
              if (abs(inv[j + result.rows * k]) >= bbb) {
                bbb = abs(inv[j + result.rows * k]);
                mRow = j;
                mCol = k;
              }
            }
          }
        }
      }
      ++(pivot[mCol]);

      if (mRow != mCol) {
        tmp = b[mRow];
        b[mRow] = b[mCol];
        b[mCol] = tmp;

        for (int v = 0; v < n; v++) {
          tmp = inv[mRow + result.rows * v];
          inv[mRow + result.rows * v] = inv[mCol + result.rows * v];
          inv[mCol + result.rows * v] = tmp;
        }
      }

      indexRow[i] = mRow;
      indexCol[i] = mCol;
      if (abs(inv[mCol + result.rows * mCol]) < 1e-16) {
        throw "Singular Matrix";
        return result;
      }
      float pivotInv = 1.0f / inv[mCol + result.rows * mCol];
      inv[mCol + result.rows * mCol] = 1.0;

      for (int v = 0; v < n; v++) {
        inv[mCol + result.rows * v] *= pivotInv;
      }
      b[mCol] *= pivotInv;
      for (int vv = 0; vv < n; vv++) {
        if (vv != mCol) {
          float dummy = inv[vv + result.rows * mCol];
          inv[vv + result.rows * mCol] = 0.0;
          for (int v = 0; v < n; v++) {
            inv[vv + result.rows * v] -= inv[mCol + result.rows * v] * dummy;
          }
          b[vv] -= b[mCol] * dummy;
        }
      }
    }

    for (int v = n - 1; v >= 0; v--) {
      if (indexRow[v] != indexCol[v]) {
        for (int k = 0; k < n; k++) {
          tmp = inv[k + result.rows * indexRow[v]];
          inv[k + result.rows * indexRow[v]] = inv[k + result.rows * indexCol[v]];
          inv[k + result.rows * indexCol[v]] = tmp;
        }
      }
    }
  }

  return result;
};