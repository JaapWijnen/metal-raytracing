//
//  Shaders.metal
//  Raytracing Shared
//
//  Created by Jaap Wijnen on 21/11/2021.
//

#include <metal_stdlib>
#include <simd/simd.h>

// Including header shared between this Metal shader code and Swift/C code executing Metal API commands

using namespace metal;
using namespace raytracing;

#include "ShaderFunctions.h"

struct TriangleResources {
    device float3 *vertexNormals;
    device float3 *vertexColors;
};

constant int resourcesStride [[function_constant(0)]];
constant int maxSubmeshes [[function_constant(1)]];

constant short constNumbers[64] = { 41, 38, 64, 45, 35, 59, 44, 11, 54, 29, 47, 26, 19, 22, 24, 7, 1, 23, 50, 9, 5, 52, 4, 56, 39, 0, 55, 25, 53, 16, 14, 13, 18, 15, 40, 60, 63, 21, 51, 30, 32, 10, 12, 33, 36, 6, 43, 57, 42, 62, 20, 28, 31, 17, 46, 34, 37, 3, 61, 58, 2, 27, 49, 8 };

struct Resource
{
    device float3 *normals;
    device int *indices;
    device Material *material;
};

[[max_total_threads_per_threadgroup(64)]]
kernel void raytracingKernel(uint2 tid [[thread_position_in_grid]],
                             uint tidtg [[ thread_index_in_threadgroup ]],
                             constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                             acceleration_structure<instancing> accelerationStructure [[ buffer(BufferIndexAccelerationStructure) ]],
                             device Resource *resources [[ buffer(BufferIndexResources) ]],
                             device MTLAccelerationStructureInstanceDescriptor *instances [[ buffer(BufferIndexInstanceDescriptors) ]],
                             device Light *lights [[ buffer(BufferIndexLights) ]],
                             texture2d<unsigned int, access::read> randomTexture [[ texture(TextureIndexRandom) ]],
                             texture2d<float, access::read> prevTex [[ texture(TextureIndexAccumulation) ]],
                             texture2d<float, access::write> dstTex [[ texture(TextureIndexPreviousAccumulation) ]])
{
    // The sample aligns the thread count to the threadgroup size. which means the thread count
    // may be different than the bounds of the texture. Test to make sure this thread
    // is referencing a pixel within the bounds of the texture.
    if ((int)tid.x < uniforms.width && (int)tid.y < uniforms.height) {
        float2 pixel = (float2)tid;
        
        // Apply a random offset to the random number index to decorrelate pixels.
        unsigned int offset = randomTexture.read(tid).x;
        
        // playing around with sorting threads here and eventually running functions batched together on the same threadgroup to reduce divergence.
        threadgroup short numbers[64];
        short number = constNumbers[tidtg];
        numbers[tidtg] = number;

        int n = 64;
        for (short k = 2; k <= n; k *= 2) { // k is doubled every iteration
            for (short j = k/2; j > 0; j /= 2) { // j is halved at every iteration, with truncation of fractional parts
                short i = tidtg;
                short l = i^j; // in C-like languages this is "i ^ j"
                if (l > i) {
                    short thing = i&k;
                    if (((thing == 0) && (numbers[i] > numbers[l])) || ((thing != 0) && (numbers[i] < numbers[l])) ) {
                        short tmp = numbers[i];
                        numbers[i] = numbers[l];
                        numbers[l] = tmp;
                    }
                }
            }
        }
        
        
        
        // Add a random offset to the pixel coordinates for antialiasing.
        float2 r = float2(halton(offset + uniforms.frameIndex, 0),
                          halton(offset + uniforms.frameIndex, 1));
        pixel += r;
        
        // Map pixel coordinates to -1..1.
        float2 uv = (float2)pixel / float2(uniforms.width, uniforms.height);
        uv = uv * 2.0 - 1.0;
        
        constant Camera & camera = uniforms.camera;
        
        ray ray;
        // Rays start at the camera position.
        ray.origin = camera.position;
        // Map normalized pixel coordinates into camera's coordinate system.
        ray.direction = normalize(uv.x * camera.right +
                                  uv.y * camera.up +
                                  camera.forward);
        // Don't limit intersection distance.
        ray.max_distance = INFINITY;
        ray.min_distance = 0;
        
        // Start with a fully white color. The kernel scales the light each time the
        // ray bounces off of a surface, based on how much of each light component
        // the surface absorbs.
        float3 color = float3(1.0f, 1.0f, 1.0f);
        float3 accumulatedColor = float3(0.0f, 0.0f, 0.0f);
        
        // Create an intersector to test for intersection between the ray and the geometry in the scene.
        intersector<triangle_data, instancing> i;
                
        i.assume_geometry_type(geometry_type::triangle);
        i.force_opacity(forced_opacity::opaque);
        
        typename intersector<triangle_data, instancing>::result_type intersection;
                
        for (int bounce = 0; bounce < 3; bounce++) {
            // Get the closest intersection, not the first intersection. This is the default, but
            // the sample adjusts this property below when it casts shadow rays.
            i.accept_any_intersection(false);
            
            // Check for intersection between the ray and the acceleration structure.
            //intersection = i.intersect(ray, accelerationStructure, bounce == 0 ? RAY_MASK_PRIMARY : RAY_MASK_SECONDARY);
            intersection = i.intersect(ray, accelerationStructure);
            // Stop if the ray didn't hit anything and has bounced out of the scene.
            if (intersection.type == intersection_type::none)
                break;
            
            int instanceIndex = intersection.instance_id;
            // only single instances currently
            int geometryIndex2 = intersection.geometry_id;
            
            // The ray hit something. Look up the transformation matrix for this instance.
            float4x4 objectToWorldSpaceTransform(1.0f);

            for (int column = 0; column < 4; column++)
                for (int row = 0; row < 3; row++)
                    objectToWorldSpaceTransform[column][row] = instances[instanceIndex].transformationMatrix[column][row];
            
            // Compute intersection point in world space.
            float3 worldSpaceIntersectionPoint = ray.origin + ray.direction * intersection.distance;
            int maxSubmeshes2 = maxSubmeshes;
            int resourceIndex = instanceIndex * maxSubmeshes2 + geometryIndex2;
            Resource resource = resources[resourceIndex];
            
            float3 objectSpaceSurfaceNormal = interpolateVertexAttribute(resource.normals, intersection, resource.indices);
            float3 worldSpaceSurfaceNormal = (objectToWorldSpaceTransform * float4(objectSpaceSurfaceNormal, 0)).xyz;
            worldSpaceSurfaceNormal = normalize(worldSpaceSurfaceNormal);
            float3 surfaceColor = resource.material->baseColor;
            
            // Choose a random light source to sample.
            float lightSample = halton(offset + uniforms.frameIndex, 2 + bounce * 5 + 0);
            int lightIndex = min((int)(lightSample * uniforms.lightCount), uniforms.lightCount - 1);
            
            device Light &light = lights[lightIndex];
            
            float3 worldSpaceLightDirection;
            float lightDistance;
            float3 lightColor;
            
            if (light.type == LightTypeAreaLight) {
                
                // Choose a random point to sample on the light source.
                r = float2(halton(offset + uniforms.frameIndex, 2 + bounce * 5 + 1),
                                  halton(offset + uniforms.frameIndex, 2 + bounce * 5 + 2));

                // Sample the lighting between the intersection point and the point on the area light.
                sampleAreaLight(light, r, worldSpaceIntersectionPoint, worldSpaceLightDirection,
                                lightColor, lightDistance);
                
                
            } else if (light.type == LightTypeSpotlight) {
                // Compute vector from sample point on light source to intersection point
                worldSpaceLightDirection = light.position - worldSpaceIntersectionPoint;
                        
                lightDistance = length(worldSpaceLightDirection);
                
                float inverseLightDistance = 1.0f / max(lightDistance, 1e-3f);
                
                // Normalize the light direction
                worldSpaceLightDirection *= inverseLightDistance;
                
//                // Start with the light's color
//                lightColor = light.color;
//
//                // Light falls off with the inverse square of the distance to the intersection point
//                lightColor *= (inverseLightDistance * inverseLightDistance);
                
                lightColor = 0.0;
                
                float3 coneDirection = normalize(light.direction);
                float spotResult = dot(-worldSpaceLightDirection, coneDirection);
                
                if (spotResult > cos(light.coneAngle)) {
                    lightColor = light.color * inverseLightDistance * inverseLightDistance;
                }
            } else if (light.type == LightTypePointlight) {
                worldSpaceLightDirection = light.position - worldSpaceIntersectionPoint;
                lightDistance = length(worldSpaceLightDirection);
                float inverseLightDistance = 1.0f / max(lightDistance, 1e-3f);
                worldSpaceLightDirection *= inverseLightDistance;
                lightColor = light.color * inverseLightDistance * inverseLightDistance;
            } else  { // light.type == LightTypeSunlight
                worldSpaceLightDirection = -normalize(light.direction);
                lightDistance = INFINITY;
                lightColor = light.color;
            }
            
            // Scale the light color by the cosine of the angle between the light direction and
            // surface normal.
            lightColor *= saturate(dot(worldSpaceSurfaceNormal, worldSpaceLightDirection));

            // Scale the light color by the number of lights to compensate for the fact that
            // the sample only samples one light source at random.
            lightColor *= uniforms.lightCount;

            // Scale the ray color by the color of the surface. This simulates light being absorbed into
            // the surface.
            color *= surfaceColor;
            
            if (length(lightColor) > 0.0001) {
            
                // Compute the shadow ray. The shadow ray checks if the sample position on the
                // light source is visible from the current intersection point.
                // If it is, the lighting contribution is added to the output image.
                struct ray shadowRay;

                // Add a small offset to the intersection point to avoid intersecting the same
                // triangle again.
                shadowRay.origin = worldSpaceIntersectionPoint + worldSpaceSurfaceNormal * 1e-3f;

                // Travel towards the light source.
                shadowRay.direction = worldSpaceLightDirection;

                // Don't overshoot the light source.
                shadowRay.max_distance = lightDistance - 1e-3f;

                // Shadow rays check only whether there is an object between the intersection point
                // and the light source. Tell Metal to return after finding any intersection.
                i.accept_any_intersection(true);

                /*if (useIntersectionFunctions)
                    intersection = i.intersect(shadowRay, accelerationStructure, RAY_MASK_SHADOW, intersectionFunctionTable);
                else
                    intersection = i.intersect(shadowRay, accelerationStructure, RAY_MASK_SHADOW);
                 */
                intersection = i.intersect(shadowRay, accelerationStructure);

                // If there was no intersection, then the light source is visible from the original
                // intersection  point. Add the light's contribution to the image.
                if (intersection.type == intersection_type::none) {
                    accumulatedColor += lightColor * color;
                }
            }

            // Next choose a random direction to continue the path of the ray. This will
            // cause light to bounce between surfaces. The sample could apply a fair bit of math
            // to compute the fraction of light reflected by the current intersection point to the
            // previous point from the next point. However, by choosing a random direction with
            // probability proportional to the cosine (dot product) of the angle between the
            // sample direction and surface normal, the math entirely cancels out except for
            // multiplying by the surface color. This sampling strategy also reduces the amount
            // of noise in the output image.
            r = float2(halton(offset + uniforms.frameIndex, 2 + bounce * 5 + 3),
                       halton(offset + uniforms.frameIndex, 2 + bounce * 5 + 4));

            float3 worldSpaceSampleDirection = sampleCosineWeightedHemisphere(r);
            worldSpaceSampleDirection = alignHemisphereWithNormal(worldSpaceSampleDirection, worldSpaceSurfaceNormal);

            ray.origin = worldSpaceIntersectionPoint + worldSpaceSurfaceNormal * 1e-3f;
            ray.direction = worldSpaceSampleDirection;
        }
                    
        // Average this frame's sample with all of the previous frames.
        if (uniforms.frameIndex > 0) {
            float3 prevColor = prevTex.read(tid).xyz;
            prevColor *= uniforms.frameIndex;

            accumulatedColor += prevColor;
            accumulatedColor /= (uniforms.frameIndex + 1);
        }

        dstTex.write(float4(accumulatedColor, 1.0f), tid);
    }
}

