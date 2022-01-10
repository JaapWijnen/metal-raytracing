//
//  Shaders.metal
//  Raytracing macOS
//
//  Created by Jaap Wijnen on 21/11/2021.
//

#include <metal_stdlib>
#include <simd/simd.h>

// Including header shared between this Metal shader code and Swift/C code executing Metal API commands
#import "ShaderTypes.h"

using namespace metal;

struct Vertex {
    float4 position [[position]];
    float2 uv;
};

constant float2 quadVertices[] = {
    float2(-1, -1),
    float2(-1,  1),
    float2( 1,  1),
    float2(-1, -1),
    float2( 1,  1),
    float2( 1, -1)
};

vertex Vertex vertexShader(unsigned short vid [[vertex_id]])
{
    float2 position = quadVertices[vid];
    Vertex out;
    out.position = float4(position, 0, 1);
    out.uv = position * 0.5 + 0.5;
    return out;
}

fragment float4 fragmentShader(Vertex in [[stage_in]],
                               texture2d<float> tex)
{
    constexpr sampler s(min_filter::nearest,
                        mag_filter::nearest,
                        mip_filter::none);
    float3 color = tex.sample(s, in.uv).xyz;
    
    // Apply a very simple tonemapping function to reduce the dynamic range of the
    // input image into a range which the screen can display.
    color = color / (1.0f + color);
    
    return float4(color, 1.0);
}
