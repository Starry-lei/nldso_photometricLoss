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
/// @ref gtx_orthonormalize
/// @file glm/gtx/orthonormalize.hpp
/// @date 2005-12-21 / 2011-06-07
/// @author Christophe Riccio
///
/// @see core (dependence)
/// @see gtx_extented_min_max (dependence)
///
/// @defgroup gtx_orthonormalize GLM_GTX_orthonormalize
/// @ingroup gtx
/// 
/// @brief Orthonormalize matrices.
/// 
/// <glm/gtx/orthonormalize.hpp> need to be included to use these functionalities.
///////////////////////////////////////////////////////////////////////////////////

#pragma once

// Dependency:
#include "gli/external/glm/glm/geometric.hpp"
#include "gli/external/glm/glm/mat3x3.hpp"
#include "gli/external/glm/glm/vec3.hpp"

#if(defined(GLM_MESSAGES) && !defined(GLM_EXT_INCLUDED))
#	pragma message("GLM: GLM_GTX_orthonormalize extension included")
#endif

namespace glm
{
	/// @addtogroup gtx_orthonormalize
	/// @{

	/// Returns the orthonormalized matrix of m.
	///
	/// @see gtx_orthonormalize
	template <typename T, precision P> 
	GLM_FUNC_DECL tmat3x3<T, P> orthonormalize(tmat3x3<T, P> const & m);
		
	/// Orthonormalizes x according y.
	///
	/// @see gtx_orthonormalize
	template <typename T, precision P> 
	GLM_FUNC_DECL tvec3<T, P> orthonormalize(tvec3<T, P> const & x, tvec3<T, P> const & y);

	/// @}
}//namespace glm

#include "orthonormalize.inl"
