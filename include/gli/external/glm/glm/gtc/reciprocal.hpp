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
/// @ref gtc_reciprocal
/// @file glm/gtc/reciprocal.hpp
/// @date 2008-10-09 / 2012-01-25
/// @author Christophe Riccio
///
/// @see core (dependence)
///
/// @defgroup gtc_reciprocal GLM_GTC_reciprocal
/// @ingroup gtc
/// 
/// @brief Define secant, cosecant and cotangent functions.
/// 
/// <glm/gtc/reciprocal.hpp> need to be included to use these features.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependencies
#include "gli/external/glm/glm/detail/setup.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTC_reciprocal extension included")
#endif

namespace glm
{
	/// @addtogroup gtc_reciprocal
	/// @{

	/// Secant function. 
	/// hypotenuse / adjacent or 1 / cos(x)
	/// 
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see gtc_reciprocal
	template <typename genType> 
	GLM_FUNC_DECL genType sec(genType angle);

	/// Cosecant function. 
	/// hypotenuse / opposite or 1 / sin(x)
	/// 
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see gtc_reciprocal
	template <typename genType> 
	GLM_FUNC_DECL genType csc(genType angle);
		
	/// Cotangent function. 
	/// adjacent / opposite or 1 / tan(x)
	/// 
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see gtc_reciprocal
	template <typename genType> 
	GLM_FUNC_DECL genType cot(genType angle);

	/// Inverse secant function. 
	/// 
	/// @return Return an angle expressed in radians.
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see gtc_reciprocal
	template <typename genType> 
	GLM_FUNC_DECL genType asec(genType x);

	/// Inverse cosecant function. 
	/// 
	/// @return Return an angle expressed in radians.
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see gtc_reciprocal
	template <typename genType> 
	GLM_FUNC_DECL genType acsc(genType x);
		
	/// Inverse cotangent function. 
	/// 
	/// @return Return an angle expressed in radians.
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see gtc_reciprocal
	template <typename genType> 
	GLM_FUNC_DECL genType acot(genType x);

	/// Secant hyperbolic function. 
	/// 
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see gtc_reciprocal
	template <typename genType> 
	GLM_FUNC_DECL genType sech(genType angle);

	/// Cosecant hyperbolic function. 
	/// 
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see gtc_reciprocal
	template <typename genType> 
	GLM_FUNC_DECL genType csch(genType angle);
		
	/// Cotangent hyperbolic function. 
	/// 
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see gtc_reciprocal
	template <typename genType> 
	GLM_FUNC_DECL genType coth(genType angle);

	/// Inverse secant hyperbolic function. 
	/// 
	/// @return Return an angle expressed in radians.
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see gtc_reciprocal
	template <typename genType> 
	GLM_FUNC_DECL genType asech(genType x);

	/// Inverse cosecant hyperbolic function. 
	/// 
	/// @return Return an angle expressed in radians.
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see gtc_reciprocal
	template <typename genType> 
	GLM_FUNC_DECL genType acsch(genType x);
		
	/// Inverse cotangent hyperbolic function. 
	/// 
	/// @return Return an angle expressed in radians.
	/// @tparam genType Floating-point scalar or vector types.
	/// 
	/// @see gtc_reciprocal
	template <typename genType> 
	GLM_FUNC_DECL genType acoth(genType x);

	/// @}
}//namespace glm

#include "reciprocal.inl"
