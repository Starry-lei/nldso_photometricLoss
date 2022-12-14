///////////////////////////////////////////////////////////////////////////////////
/// OpenGL Mathematics (glm.g-truc.net)
///
/// Copyright (c) 2005 - 2015 G-Truc Creation (www.g-truc.net)
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
/// 
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// 
/// Restrictions:
///		By making use of the Software for military purposes, you choose to make
///		a Bunny unhappy.
/// 
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
/// THE SOFTWARE.
///
/// @file glm/glm.hpp
/// @date 2009-05-01 / 2011-05-16
/// @author Christophe Riccio
///
/// @ref core (Dependence)
/// 
/// @defgroup gtc GTC Extensions (Stable)
///
/// @brief Functions and types that the GLSL specification doesn't define, but useful to have for a C++ program.
/// 
/// GTC extensions aim to be stable. 
/// 
/// Even if it's highly unrecommended, it's possible to include all the extensions at once by
/// including <glm/ext.hpp>. Otherwise, each extension needs to be included  a specific file.
/// 
/// @defgroup gtx GTX Extensions (Experimental)
/// 
/// @brief Functions and types that the GLSL specification doesn't define, but 
/// useful to have for a C++ program.
/// 
/// Experimental extensions are useful functions and types, but the development of
/// their API and functionality is not necessarily stable. They can change 
/// substantially between versions. Backwards compatibility is not much of an issue
/// for them.
/// 
/// Even if it's highly unrecommended, it's possible to include all the extensions 
/// at once by including <glm/ext.hpp>. Otherwise, each extension needs to be 
/// included  a specific file.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

#if(defined(GLM_MESSAGES) && !defined(GLM_MESSAGE_EXT_INCLUDED_DISPLAYED))
#	define GLM_MESSAGE_EXT_INCLUDED_DISPLAYED
#	pragma message("GLM: All extensions included (not recommanded)")
#endif//GLM_MESSAGES

#include "gli/external/glm/glm/gtc/bitfield.hpp"
#include "gli/external/glm/glm/gtc/constants.hpp"
#include "gli/external/glm/glm/gtc/epsilon.hpp"
#include "gli/external/glm/glm/gtc/integer.hpp"
#include "gli/external/glm/glm/gtc/matrix_access.hpp"
#include "gli/external/glm/glm/gtc/matrix_integer.hpp"
#include "gli/external/glm/glm/gtc/matrix_inverse.hpp"
#include "gli/external/glm/glm/gtc/matrix_transform.hpp"
#include "gli/external/glm/glm/gtc/noise.hpp"
#include "gli/external/glm/glm/gtc/packing.hpp"
#include "gli/external/glm/glm/gtc/quaternion.hpp"
#include "gli/external/glm/glm/gtc/random.hpp"
#include "gli/external/glm/glm/gtc/reciprocal.hpp"
#include "gli/external/glm/glm/gtc/round.hpp"
#include "gli/external/glm/glm/gtc/type_precision.hpp"
#include "gli/external/glm/glm/gtc/type_ptr.hpp"
#include "gli/external/glm/glm/gtc/ulp.hpp"
#include "gli/external/glm/glm/gtc/vec1.hpp"

#include "gli/external/glm/glm/gtx/associated_min_max.hpp"
#include "gli/external/glm/glm/gtx/bit.hpp"
#include "gli/external/glm/glm/gtx/closest_point.hpp"
#include "gli/external/glm/glm/gtx/color_space.hpp"
#include "gli/external/glm/glm/gtx/color_space_YCoCg.hpp"
#include "gli/external/glm/glm/gtx/compatibility.hpp"
#include "gli/external/glm/glm/gtx/component_wise.hpp"
#include "gli/external/glm/glm/gtx/dual_quaternion.hpp"
#include "gli/external/glm/glm/gtx/euler_angles.hpp"
#include "gli/external/glm/glm/gtx/extend.hpp"
#include "gli/external/glm/glm/gtx/extended_min_max.hpp"
#include "gli/external/glm/glm/gtx/fast_exponential.hpp"
#include "gli/external/glm/glm/gtx/fast_square_root.hpp"
#include "gli/external/glm/glm/gtx/fast_trigonometry.hpp"
#include "gli/external/glm/glm/gtx/gradient_paint.hpp"
#include "gli/external/glm/glm/gtx/handed_coordinate_space.hpp"
#include "gli/external/glm/glm/gtx/integer.hpp"
#include "gli/external/glm/glm/gtx/intersect.hpp"
#include "gli/external/glm/glm/gtx/log_base.hpp"
#include "gli/external/glm/glm/gtx/matrix_cross_product.hpp"
#include "gli/external/glm/glm/gtx/matrix_interpolation.hpp"
#include "gli/external/glm/glm/gtx/matrix_major_storage.hpp"
#include "gli/external/glm/glm/gtx/matrix_operation.hpp"
#include "gli/external/glm/glm/gtx/matrix_query.hpp"
#include "gli/external/glm/glm/gtx/mixed_product.hpp"
#include "gli/external/glm/glm/gtx/norm.hpp"
#include "gli/external/glm/glm/gtx/normal.hpp"
#include "gli/external/glm/glm/gtx/normalize_dot.hpp"
#include "gli/external/glm/glm/gtx/number_precision.hpp"
#include "gli/external/glm/glm/gtx/optimum_pow.hpp"
#include "gli/external/glm/glm/gtx/orthonormalize.hpp"
#include "gli/external/glm/glm/gtx/perpendicular.hpp"
#include "gli/external/glm/glm/gtx/polar_coordinates.hpp"
#include "gli/external/glm/glm/gtx/projection.hpp"
#include "gli/external/glm/glm/gtx/quaternion.hpp"
#include "gli/external/glm/glm/gtx/raw_data.hpp"
#include "gli/external/glm/glm/gtx/rotate_vector.hpp"
#include "gli/external/glm/glm/gtx/spline.hpp"
#include "gli/external/glm/glm/gtx/std_based_type.hpp"
#if !(GLM_COMPILER & GLM_COMPILER_CUDA)
#	include "gli/external/glm/glm/gtx/string_cast.hpp"
#endif
#include "gli/external/glm/glm/gtx/transform.hpp"
#include "gli/external/glm/glm/gtx/transform2.hpp"
#include "gli/external/glm/glm/gtx/type_aligned.hpp"
#include "gli/external/glm/glm/gtx/vector_angle.hpp"
#include "gli/external/glm/glm/gtx/vector_query.hpp"
#include "gli/external/glm/glm/gtx/wrap.hpp"

#if GLM_HAS_TEMPLATE_ALIASES
#	include "gli/external/glm/glm/gtx/scalar_multiplication.hpp"
#endif

#if GLM_HAS_RANGE_FOR
#	include "gli/external/glm/glm/gtx/range.hpp"
#endif

#if GLM_ARCH & GLM_ARCH_SSE2
#	include "gli/external/glm/glm/gtx/simd_vec4.hpp"
#	include "gli/external/glm/glm/gtx/simd_mat4.hpp"
#endif
