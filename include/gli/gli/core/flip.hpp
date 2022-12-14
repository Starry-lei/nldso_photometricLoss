#pragma once

#include "gli/gli/texture2d.hpp"
#include "gli/gli/texture2d_array.hpp"

namespace gli
{
	template <typename texture>
	texture flip(texture const & Texture);

}//namespace gli

#include "flip.inl"
