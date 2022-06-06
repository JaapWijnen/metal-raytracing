//
//  RTShaders.metal
//  MetalRaytracing
//
//  Created by Jaap Wijnen on 20/04/2022.
//

#include <metal_stdlib>
using namespace metal;
using namespace raytracing;

#include "ShaderFunctions.h"

#define BINNING_TILE_SIZE 4
#define MAX_BINNING_TILE_SIZE_INDEX (BINNING_TILE_SIZE * BINNING_TILE_SIZE)

float2 octWrap( float2 v )
{
    return (1.0 - abs(v.yx)) * float2((v.x >= 0.0 ? 1.0 : -1.0), v.y >= 0.0 ? 1.0 : -1.0);
}
 
float2 encodeOct( float3 n )
{
    n /= ( abs( n.x ) + abs( n.y ) + abs( n.z ) );
    n.xy = n.z >= 0.0 ? n.xy : octWrap( n.xy );
    n.xy = n.xy * 0.5 + 0.5;
    return n.xy;
}
 
float3 decodeOct( float2 f )
{
    f = f * 2.0 - 1.0;
 
    // https://twitter.com/Stubbesaurus/status/937994790553227264
    float3 n = float3( f.x, f.y, 1.0 - abs( f.x ) - abs( f.y ) );
    float t = saturate( -n.z );
    n.xy += float2(n.x >= 0.0 ? -t : t, n.y >= 0.0 ? -t : t);
    return normalize( n );
}

kernel void rb_ray_bin_kernel(uint2 tid [[ thread_position_in_grid ]],
                              uint2 ttgid [[ thread_position_in_threadgroup ]],
                              uint2 tgid [[ threadgroup_position_in_grid ]],
                              texture2d<float, access::read> rayDirections [[ texture(TextureIndexRayDirections) ]],
                              constant float *rayMaxDistances [[ buffer(BufferIndexRayMaxDistances) ]],
                              device uint2 *rayBinningCoordinates [[ buffer(BufferIndexRayBinningCoordinates) ]],
                              //device atomic_int *binSizes [[ buffer(BufferIndexRayBinSizes) ]],
                              device int *raysPerTile [[ buffer(BufferIndexRaysPerTile) ]],
                              constant Uniforms &uniforms [[ buffer(BufferIndexUniforms) ]])
{
    uint binIndex;
    threadgroup atomic_int binSize[MAX_BINNING_TILE_SIZE_INDEX];
    int rayBinIndex;
    if ((int)tid.x < uniforms.width && (int)tid.y < uniforms.height) {
        float3 rayDirection = rayDirections.read(tid).xyz;
        float2 octEncoded = encodeOct(rayDirection);
        uint2 binCoord = uint2((BINNING_TILE_SIZE - 1) * octEncoded);
        binIndex = binCoord.y * BINNING_TILE_SIZE + binCoord.x;

        uint allBinsIndex = ttgid.y * BINNING_TILE_SIZE + ttgid.x;
        atomic_store_explicit(&binSize[allBinsIndex], 0, memory_order_relaxed);
        
        rayBinIndex = atomic_fetch_add_explicit(&binSize[binIndex], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
        
    threadgroup int binOffset[MAX_BINNING_TILE_SIZE_INDEX];
    if (ttgid.x == 0 && ttgid.y == 0) {
        binOffset[0] = 0;
        for (int i = 0; i < MAX_BINNING_TILE_SIZE_INDEX; ++i) {
            int binSizeValue = atomic_load_explicit(&binSize[i - 1], memory_order_relaxed);
            binOffset[i] = binOffset[i - 1] + binSizeValue;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if ((int)tid.x < uniforms.width && (int)tid.y < uniforms.height) {

        uint binTileCountX = (uniforms.width + BINNING_TILE_SIZE - 1) / BINNING_TILE_SIZE;
        uint groupdIndex = tgid.y * binTileCountX + tgid.x;
        uint tileOffset = groupdIndex * MAX_BINNING_TILE_SIZE_INDEX;
        uint binOffsetValue = binOffset[binIndex];
        
        uint offset = tileOffset + binOffsetValue + rayBinIndex;
        rayBinningCoordinates[offset] = tid;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ttgid.x == 0 && ttgid.y == 0) {
        uint binTileCountX = (uniforms.width + BINNING_TILE_SIZE - 1) / BINNING_TILE_SIZE;
        uint groupdIndex = tgid.y * binTileCountX + tgid.x;
        int binSizeValue = atomic_load_explicit(&binSize[MAX_BINNING_TILE_SIZE_INDEX - 1], memory_order_relaxed);
        raysPerTile[groupdIndex] = binOffset[MAX_BINNING_TILE_SIZE_INDEX - 1] + binSizeValue;
    }
    
    if ((int)tid.x < uniforms.width && (int)tid.y < uniforms.height) {
        // Compute the coordinates of the bin within each tile
        uint2 perTileCoord = uint2(tid.x % BINNING_TILE_SIZE, tid.y % BINNING_TILE_SIZE);
        uint perTileIndex = perTileCoord.y * BINNING_TILE_SIZE + perTileCoord.x;

        // Get the tile coordinates
        uint2 tileCoord = uint2(tid.x / BINNING_TILE_SIZE, tid.y / BINNING_TILE_SIZE);
        uint binTileCountX = (uniforms.width + BINNING_TILE_SIZE - 1) / BINNING_TILE_SIZE;
        uint tileIndex = tileCoord.y * binTileCountX + tileCoord.x;

        // Get the screen space coordinates for this bin
        uint binIndexReturn = tileIndex * BINNING_TILE_SIZE * BINNING_TILE_SIZE + perTileIndex ;
        uint2 binCoordReturn = rayBinningCoordinates[binIndexReturn];

        //Read the ray direction
        float3 rayDirectionReturn = rayDirections.read(binCoordReturn).xyz;
    }
}
