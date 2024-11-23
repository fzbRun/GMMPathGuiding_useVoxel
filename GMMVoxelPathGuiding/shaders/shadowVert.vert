#version 450

layout(location = 0) in vec4 inPos;

layout(binding = 0) uniform UniformBufferObject{
	mat4 model;
	mat4 view;
	mat4 proj;
	vec4 lightPos_strength;
	vec4 normal;
	vec4 size;
}ubo;

void main(){
	gl_Position = ubo.proj * ubo.view * ubo.model * inPos;
}