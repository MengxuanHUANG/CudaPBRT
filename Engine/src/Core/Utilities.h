#pragma once

#include <ranges>
#include <vector>
#include <string_view>

namespace CudaPBRT
{
	namespace StringUtility
	{
		inline std::vector<std::string> Split(const std::string& str, const std::string& delim)
		{
			const static auto transform_func = [](auto word) {
				return std::string(word.begin(), word.end());
			};

			auto v = str | std::views::split(delim) | std::views::transform(transform_func);

			return { v.begin(), v.end() };
		}
	}
}