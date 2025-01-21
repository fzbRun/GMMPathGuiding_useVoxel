import numpy as np
import glm

def packUnorm4x8(value):
    value = glm.ivec4(glm.clamp(value * 128, 0.0, 255.0))
    return (int(value.w)&0x000000FF) << 24 | (int(value.z)&0x000000FF) << 16 | (int(value.y)&0x000000FF) << 8 | (int(value.x)&0x000000FF)

def unPackUnorm4x8(value):
    return glm.vec4(float((value&0x000000FF)), float((value&0x0000FF00) >> 8), float((value&0x00FF0000) >> 16), float((value&0xFF000000) >> 24))

def packUnorm2x16(value):
    value = glm.ivec2(glm.clamp(value * 32768, 0.0, 65535.0))
    return (int(value.y)&0x0000FFFF) << 16 | (int(value.x)&0x0000FFFF)

def unPackUnorm2x16(value):
    return glm.vec2(float((value&0x0000FFFF)), float((value&0xFFFF0000) >> 16))

#random = glm.vec4(1.1, 0.04, 2.0, 0.001)
#random_uint = packUnorm4x8(random)
#random = unPackUnorm4x8(random_uint)
#print(random_uint)
#print(random / 128.0)

random = glm.vec2(1.1, 0.00004)
random_uint = packUnorm2x16(random)
random = unPackUnorm2x16(random_uint)
print(random_uint)
print(random / 32768.0)
