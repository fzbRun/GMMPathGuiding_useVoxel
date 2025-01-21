C:/D/Vulkan/Bin/glslc.exe -fshader-stage=compute updateSS.glsl -o updateSS.spv
C:/D/Vulkan/Bin/glslc.exe -fshader-stage=compute -I ./ updateGMMPara.glsl -o updateGMMPara.spv
C:/D/Vulkan/Bin/glslc.exe -fshader-stage=compute -I ./ pathGuiding.glsl -o pathGuiding.spv
pause