#include <gli/gli.hpp>
#include <ctime>

namespace clear
{
	template <typename texture, typename genType>
	int test_texture(typename texture::extent_type Size, gli::format const & Format, std::vector<genType> const & Colors)
	{
		int Error(0);

		texture TextureA(Format, Size);

		for(typename texture::size_type Level = 0; Level < TextureA.levels(); ++Level)
			TextureA[Level].clear(Colors[Level]);

		texture TextureB = TextureA;
		Error += TextureB == TextureA ? 0 : 1;

		texture TextureC(gli::duplicate(TextureA));
		Error += TextureC == TextureA ? 0 : 1;

		texture TextureD(gli::duplicate(TextureB));
		Error += TextureD == TextureB ? 0 : 1;

		texture TextureE(gli::duplicate(TextureC, 1, 2));
		texture TextureF(TextureC, 1, 2);
		Error += TextureE == TextureF ? 0 : 1;

		texture TextureG(gli::duplicate(TextureD, 1, 2));
		texture TextureH(TextureD, 1, 2);
		Error += TextureG == TextureH ? 0 : 1;

		TextureG.clear(Colors[Colors.size() - 1]);
		TextureH.clear(Colors[Colors.size() - 1]);
		Error += TextureG == TextureH ? 0 : 1;

		TextureG.clear();
		TextureH.clear();
		Error += TextureG == TextureH ? 0 : 1;

		return Error;
	}

	int test()
	{
		std::vector<glm::u64> ColorDXT1;
		ColorDXT1.push_back(255);
		ColorDXT1.push_back(127);
		ColorDXT1.push_back(64);
		ColorDXT1.push_back(32);
		ColorDXT1.push_back(16);
		ColorDXT1.push_back(8);
		ColorDXT1.push_back(4);

		std::vector<glm::u8> ColorR8_UNORM;
		ColorR8_UNORM.push_back(255);
		ColorR8_UNORM.push_back(127);
		ColorR8_UNORM.push_back(64);
		ColorR8_UNORM.push_back(32);
		ColorR8_UNORM.push_back(16);
		ColorR8_UNORM.push_back(8);
		ColorR8_UNORM.push_back(4);

		std::vector<glm::u8vec3> ColorRGB8_UNORM;
		ColorRGB8_UNORM.push_back(glm::u8vec3(255,   0,   0));
		ColorRGB8_UNORM.push_back(glm::u8vec3(255, 127,   0));
		ColorRGB8_UNORM.push_back(glm::u8vec3(255, 255,   0));
		ColorRGB8_UNORM.push_back(glm::u8vec3(  0, 255,   0));
		ColorRGB8_UNORM.push_back(glm::u8vec3(  0, 255, 255));
		ColorRGB8_UNORM.push_back(glm::u8vec3(  0,   0, 255));
		ColorRGB8_UNORM.push_back(glm::u8vec3(255,   0, 255));

		std::vector<glm::u8vec4> ColorRGBA8_UNORM;
		ColorRGBA8_UNORM.push_back(glm::u8vec4(255,   0,   0, 255));
		ColorRGBA8_UNORM.push_back(glm::u8vec4(255, 127,   0, 255));
		ColorRGBA8_UNORM.push_back(glm::u8vec4(255, 255,   0, 255));
		ColorRGBA8_UNORM.push_back(glm::u8vec4(  0, 255,   0, 255));
		ColorRGBA8_UNORM.push_back(glm::u8vec4(  0, 255, 255, 255));
		ColorRGBA8_UNORM.push_back(glm::u8vec4(  0,   0, 255, 255));
		ColorRGBA8_UNORM.push_back(glm::u8vec4(255,   0, 255, 255));

		std::vector<glm::vec4> ColorRGBA32F;
		ColorRGBA32F.push_back(glm::vec4(1.0,   0,   0, 1.0));
		ColorRGBA32F.push_back(glm::vec4(1.0, 0.5,   0, 1.0));
		ColorRGBA32F.push_back(glm::vec4(1.0, 1.0,   0, 1.0));
		ColorRGBA32F.push_back(glm::vec4(  0, 1.0,   0, 1.0));
		ColorRGBA32F.push_back(glm::vec4(  0, 1.0, 1.0, 1.0));
		ColorRGBA32F.push_back(glm::vec4(  0,   0, 1.0, 1.0));
		ColorRGBA32F.push_back(glm::vec4(1.0,   0, 1.0, 1.0));

		std::vector<int> Sizes;
		Sizes.push_back(32);
		Sizes.push_back(16);
		Sizes.push_back(17);
		Sizes.push_back(15);
		Sizes.push_back(12);
		Sizes.push_back(5);

		int Error(0);

		for(std::size_t i = 0; i < Sizes.size(); ++i)
		{
			//Error += test_texture<gli::texture1d>(gli::texture1d::dim_type(Sizes[i]), gli::FORMAT_RGB_DXT1_UNORM_BLOCK8, ColorDXT1);
			Error += test_texture<gli::texture2d>(gli::texture2d::extent_type(Sizes[i]), gli::FORMAT_RGB_DXT1_UNORM_BLOCK8, ColorDXT1);
			Error += test_texture<gli::texture3d>(gli::texture3d::extent_type(Sizes[i]), gli::FORMAT_RGB_DXT1_UNORM_BLOCK8, ColorDXT1);
			Error += test_texture<gli::texture1d>(gli::texture1d::extent_type(Sizes[i]), gli::FORMAT_R8_UNORM_PACK8, ColorR8_UNORM);
			Error += test_texture<gli::texture2d>(gli::texture2d::extent_type(Sizes[i]), gli::FORMAT_R8_UNORM_PACK8, ColorR8_UNORM);
			Error += test_texture<gli::texture3d>(gli::texture3d::extent_type(Sizes[i]), gli::FORMAT_R8_UNORM_PACK8, ColorR8_UNORM);
			Error += test_texture<gli::texture1d>(gli::texture1d::extent_type(Sizes[i]), gli::FORMAT_RGB8_UNORM_PACK8, ColorRGB8_UNORM);
			Error += test_texture<gli::texture2d>(gli::texture2d::extent_type(Sizes[i]), gli::FORMAT_RGB8_UNORM_PACK8, ColorRGB8_UNORM);
			Error += test_texture<gli::texture3d>(gli::texture3d::extent_type(Sizes[i]), gli::FORMAT_RGB8_UNORM_PACK8, ColorRGB8_UNORM);
			Error += test_texture<gli::texture1d>(gli::texture1d::extent_type(Sizes[i]), gli::FORMAT_RGBA8_UNORM_PACK8, ColorRGBA8_UNORM);
			Error += test_texture<gli::texture2d>(gli::texture2d::extent_type(Sizes[i]), gli::FORMAT_RGBA8_UNORM_PACK8, ColorRGBA8_UNORM);
			Error += test_texture<gli::texture3d>(gli::texture3d::extent_type(Sizes[i]), gli::FORMAT_RGBA8_UNORM_PACK8, ColorRGBA8_UNORM);
			Error += test_texture<gli::texture1d>(gli::texture1d::extent_type(Sizes[i]), gli::FORMAT_RGBA32_SFLOAT_PACK32, ColorRGBA32F);
			Error += test_texture<gli::texture2d>(gli::texture2d::extent_type(Sizes[i]), gli::FORMAT_RGBA32_SFLOAT_PACK32, ColorRGBA32F);
			Error += test_texture<gli::texture3d>(gli::texture3d::extent_type(Sizes[i]), gli::FORMAT_RGBA32_SFLOAT_PACK32, ColorRGBA32F);
		}

		return Error;
	}
}//namespace clear

int main()
{
	int Error(0);

	Error += clear::test();

	return Error;
}
