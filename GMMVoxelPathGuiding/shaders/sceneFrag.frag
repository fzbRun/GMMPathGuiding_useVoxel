#version 450

layout(location = 0) in vec2 texCoord;

layout(binding = 0) uniform sampler2D pathTracingMap;

layout(location = 0) out vec4 outColor;

void main(){
    outColor = texture(pathTracingMap, texCoord);
}