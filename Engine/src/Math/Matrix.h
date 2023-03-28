#pragma once

#include <glm/glm.hpp>

// a wrapper for glm::mat3, glm::mat4

class mat3
{
public:
	mat3(const float& f)
		:m(f)
	{}
	mat3(const glm::mat3& m)
		:m(m)
	{}

	mat3 operator*(const mat3& m2) const
	{
		return { m2.m * m };
	}

	glm::vec3 operator*(const glm::vec3& v) const
	{
		return m * v;
	}

public:
	glm::mat3 m;
};

class mat4
{
public:
	mat4(const float& f)
		:m(f)
	{}
	mat4(const glm::mat4& m)
		:m(m)
	{}

	mat4 operator*(const mat4& m2) const
	{
		return { m2.m * m };
	}

	glm::vec4 operator*(const glm::vec4& v) const
	{
		return m * v;
	}

public:
	glm::mat4 m;
};