#pragma once

#include <string>
#include <vector>
#include <ranges>
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

		inline constexpr auto hash_djb2a(const std::string_view sv) {
			unsigned long hash{ 43933 };
			for (unsigned char c : sv) {
				hash = ((hash << 6) + hash) ^ c;
			}
			return hash;
		}
		
		inline constexpr auto operator"" _sh(const char* str, size_t len) {
			return hash_djb2a(std::string_view{ str, len });
		}
	}
}