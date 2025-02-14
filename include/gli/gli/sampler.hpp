/// @brief Include to use wrap modes and the sampler base class.
/// @file gli/sampler.hpp

#pragma once

#include "filter.hpp"

namespace gli
{
	/// Texture coordinate wrapping mode
	enum wrap
	{
		WRAP_CLAMP_TO_EDGE, WRAP_FIRST = WRAP_CLAMP_TO_EDGE,
		WRAP_CLAMP_TO_BORDER,
		WRAP_REPEAT,
		WRAP_MIRROR_REPEAT,
		WRAP_MIRROR_CLAMP_TO_EDGE,
		WRAP_MIRROR_CLAMP_TO_BORDER, WRAP_LAST = WRAP_MIRROR_CLAMP_TO_BORDER
	};

	enum
	{
		WRAP_COUNT = WRAP_LAST - WRAP_FIRST + 1
	};

	/// Evaluate whether the texture coordinate wrapping mode relies on border color
	inline bool is_border(wrap Wrap)
	{
		return Wrap == WRAP_CLAMP_TO_BORDER || Wrap == WRAP_MIRROR_CLAMP_TO_BORDER;
	}

	/// Genetic sampler class.
	class sampler
	{
	public:
		sampler(wrap Wrap, filter Mip, filter Min);

	protected:
		typedef float(*wrap_type)(float const & SamplerCoord);

		wrap_type getFunc(wrap WrapMode) const;

		wrap_type Wrap;
		filter Mip;
		filter Min;
	};
}//namespace gli

#include "gli/gli/core/sampler.inl"
