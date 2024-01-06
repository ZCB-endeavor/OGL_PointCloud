#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 ourColor; // 向片段着色器输出一个颜色

void main()
{
	gl_Position = projection * view * model * vec4(aPos, 1.0f);
	//gl_Position = vec4(aPos, 1.0f);
	ourColor = aColor;
}