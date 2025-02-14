/// @brief Include to use generic textures which can represent any texture target but they don't have target specific built-in caches making accesses slower.
/// @file gli/texture.hpp

#pragma once

#include "image.hpp"
#include "target.hpp"

namespace gli
{
	/// Genetic texture class. It can support any target.
	class texture
	{
	public:
		typedef size_t size_type;
		typedef gli::target target_type;
		typedef gli::format format_type;
		typedef gli::swizzles swizzles_type;
		typedef storage::data_type data_type;
		typedef storage::extent_type extent_type;

		/// Create an empty texture instance
		texture();

		/// Create a texture object and allocate a texture storoge for it
		/// @param Target Type/Shape of the texture storage
		/// @param Format Texel format
		/// @param Extent Size of the texture: width, height and depth.
		/// @param Layers Number of one-dimensional or two-dimensional images of identical size and format
		/// @param Faces 6 for cube map textures otherwise 1.
		/// @param Levels Number of images in the texture mipmap chain.
		/// @param Swizzles A mechanism to swizzle the components of a texture before they are applied according to the texture environment.
		texture(
			target_type Target,
			format_type Format,
			extent_type const& Extent,
			size_type Layers,
			size_type Faces,
			size_type Levels,
			swizzles_type const& Swizzles = swizzles_type(SWIZZLE_RED, SWIZZLE_GREEN, SWIZZLE_BLUE, SWIZZLE_ALPHA));

		/// Create a texture object by sharing an existing texture storage from another texture instance.
		/// This texture object is effectively a texture view where the layer, the face and the level allows identifying
		/// a specific subset of the texture storage source. 
		/// This texture object is effectively a texture view where the target and format can be reinterpreted
		/// with a different compatible texture target and texture format.
		texture(
			texture const& Texture,
			target_type Target,
			format_type Format,
			size_type BaseLayer, size_type MaxLayer,
			size_type BaseFace, size_type MaxFace,
			size_type BaseLevel, size_type MaxLevel,
			swizzles_type const& Swizzles = swizzles_type(SWIZZLE_RED, SWIZZLE_GREEN, SWIZZLE_BLUE, SWIZZLE_ALPHA));

		/// Create a texture object by sharing an existing texture storage from another texture instance.
		/// This texture object is effectively a texture view where the target and format can be reinterpreted
		/// with a different compatible texture target and texture format.
		texture(
			texture const& Texture,
			target_type Target,
			format_type Format,
			swizzles_type const& Swizzles = swizzles_type(SWIZZLE_RED, SWIZZLE_GREEN, SWIZZLE_BLUE, SWIZZLE_ALPHA));

		virtual ~texture(){}

		/// Return whether the texture instance is empty, no storage or description have been assigned to the instance.
		bool empty() const;

		/// Return the target of a texture instance. 
		target_type target() const{return this->Target;}

		/// Return the texture instance format
		format_type format() const;

		swizzles_type swizzles() const;

		/// Return the base layer of the texture instance, effectively a memory offset in the actual texture storage to identify where to start reading the layers. 
		size_type base_layer() const;

		/// Return the max layer of the texture instance, effectively a memory offset to the beginning of the last layer in the actual texture storage that the texture instance can access. 
		size_type max_layer() const;

		/// Return max_layer() - base_layer() + 1
		size_type layers() const;

		/// Return the base face of the texture instance, effectively a memory offset in the actual texture storage to identify where to start reading the faces. 
		size_type base_face() const;

		/// Return the max face of the texture instance, effectively a memory offset to the beginning of the last face in the actual texture storage that the texture instance can access. 
		size_type max_face() const;

		/// Return max_face() - base_face() + 1
		size_type faces() const;

		/// Return the base level of the texture instance, effectively a memory offset in the actual texture storage to identify where to start reading the levels. 
		size_type base_level() const;

		/// Return the max level of the texture instance, effectively a memory offset to the beginning of the last level in the actual texture storage that the texture instance can access. 
		size_type max_level() const;

		/// Return max_level() - base_level() + 1.
		size_type levels() const;

		/// Return the size of a texture instance: width, height and depth.
		extent_type extent(size_type Level = 0) const;

		/// Return the memory size of a texture instance storage in bytes.
		size_type size() const;

		/// Return the number of blocks contained in a texture instance storage.
		/// genType size must match the block size conresponding to the texture format.
		template <typename genType>
		size_type size() const;

		/// Return the memory size of a specific level identified by Level.
		size_type size(size_type Level) const;

		/// Return the memory size of a specific level identified by Level.
		/// genType size must match the block size conresponding to the texture format.
		template <typename genType>
		size_type size(size_type Level) const;

		/// Return a pointer to the beginning of the texture instance data.
		void* data();

		/// Return a pointer of type genType which size must match the texture format block size
		template <typename genType>
		genType* data();

		/// Return a pointer to the beginning of the texture instance data.
		void const* data() const;

		/// Return a pointer of type genType which size must match the texture format block size
		template <typename genType>
		genType const* data() const;

		/// Return a pointer to the beginning of the texture instance data.
		void* data(size_type Layer, size_type Face, size_type Level);

		/// Return a pointer to the beginning of the texture instance data.
		void const* data(size_type Layer, size_type Face, size_type Level) const;

		/// Return a pointer of type genType which size must match the texture format block size
		template <typename genType>
		genType* data(size_type Layer, size_type Face, size_type Level);

		/// Return a pointer of type genType which size must match the texture format block size
		template <typename genType>
		genType const* data(size_type Layer, size_type Face, size_type Level) const;

		/// Clear the entire texture storage with zeros
		void clear();

		/// Clear the entire texture storage with Texel which type must match the texture storage format block size
		/// If the type of genType doesn't match the type of the texture format, no conversion is performed and the data will be reinterpreted as if is was of the texture format. 
		template <typename genType>
		void clear(genType const & Texel);

		/// Clear a specific image of a texture.
		template <typename genType>
		void clear(size_type Layer, size_type Face, size_type Level, genType const & Texel);

		/// Reorder the component in texture memory.
		template <typename genType>
		void swizzle(gli::swizzles const & Swizzles);

	protected:
		/// Compute the relative memory offset to access the data for a specific layer, face and level
		size_type base_offset(size_type Layer, size_type Face, size_type Level) const;

		struct cache
		{
			data_type* Data;
			size_type Size;
		};

		std::shared_ptr<storage> Storage;
		target_type const Target;
		format_type const Format;
		size_type const BaseLayer;
		size_type const MaxLayer;
		size_type const BaseFace;
		size_type const MaxFace;
		size_type const BaseLevel;
		size_type const MaxLevel;
		swizzles_type const Swizzles;
		cache Cache;

	private:
		void build_cache();
	};

}//namespace gli

#include "gli/gli/core/texture.inl"
