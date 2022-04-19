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
using namespace raytracing;

constant int resourcesStride [[function_constant(0)]];
constant int maxSubmeshes [[function_constant(1)]];

struct Vertex {
    float4 position [[position]];
    float2 uv;
};

constant short primes2[] = {
    2,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,  37,  41,  43,  47,  53,  59,  61,  67,  71,
    73,  79,  83,  89,  97,  101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
    419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541
};

half halton2(int i, short d) {
    short b = primes2[d];

    half f = 1.0f;
    half invB = 1.0f / b;

    half r = 0;

    while (i > 0) {
        f = f * invB;
        r = r + f * (i % b);
        i = i / b;
    }

    return r;
}

inline float3 alignHemisphereWithNormal2(float3 sample, float3 normal) {
    // Set the "up" vector to the normal
    float3 up = normal;
    
    // Find an arbitrary direction perpendicular to the normal. This will become the
    // "right" vector.
    float3 right = normalize(cross(normal, float3(0.0072f, 1.0f, 0.0034f)));
    
    // Find a third vector perpendicular to the previous two. This will be the
    // "forward" vector.
    float3 forward = cross(right, up);
    
    // Map the direction on the unit hemisphere to the coordinate system aligned
    // with the normal.
    return sample.x * right + sample.y * up + sample.z * forward;
}

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

inline void sampleAreaLight2(device Light & light,
                            float2 u,
                            float3 position,
                            thread float3 & lightDirection,
                            thread float3 & lightColor,
                            thread float & lightDistance)
{
    // Map to -1..1
    u = u * 2.0f - 1.0f;
    
    // Transform into light's coordinate system
    float3 samplePosition = light.position +
    light.right * u.x +
    light.up * u.y;
    
    // Compute vector from sample point on light source to intersection point
    lightDirection = samplePosition - position;
    
    lightDistance = length(lightDirection);
    
    float inverseLightDistance = 1.0f / max(lightDistance, 1e-3f);
    
    // Normalize the light direction
    lightDirection *= inverseLightDistance;
    
    // Start with the light's color
    lightColor = light.color;
    
    // Light falls off with the inverse square of the distance to the intersection point
    lightColor *= (inverseLightDistance * inverseLightDistance);
    
    // Light also falls off with the cosine of angle between the intersection point and
    // the light source
    lightColor *= saturate(dot(-lightDirection, light.forward));
}


// Interpolates vertex attribute of an arbitrary type across the surface of a triangle
// given the barycentric coordinates and triangle index in an intersection struct
template<typename T>
inline T interpolateVertexAttribute2(device T *attributes, intersector<triangle_data, instancing>::result_type intersection, device int *vertexIndices) {
    float3 uvw;
    uvw.xy = intersection.triangle_barycentric_coord;
    uvw.z = 1.0 - uvw.x - uvw.y;
    unsigned int triangleIndex = intersection.primitive_id;
    unsigned int index1 = vertexIndices[triangleIndex * 3 + 1];
    unsigned int index2 = vertexIndices[triangleIndex * 3 + 2];
    unsigned int index3 = vertexIndices[triangleIndex * 3 + 0];
    T T0 = attributes[index1];
    T T1 = attributes[index2];
    T T2 = attributes[index3];
    return uvw.x * T0 + uvw.y * T1 + uvw.z * T2;
}

inline float3 sampleCosineWeightedHemisphere2(float2 u) {
    float phi = 2.0f * M_PI_F * u.x;
    
    float cos_phi;
    float sin_phi = sincos(phi, cos_phi);
    
    float cos_theta = sqrt(u.y);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    
    return float3(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
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

vertex Vertex raytracingVertex(unsigned short vid [[vertex_id]])
{
    float2 position = quadVertices[vid];
    Vertex out;
    out.position = float4(position, 0, 1);
    out.uv = position * 0.5 + 0.5;
    return out;
}

struct Resource
{
    device float3 *normals;
    device int *indices;
    device Material *material;
};

fragment float4 raytracingFragment(Vertex in [[stage_in]],
                                   constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                                   acceleration_structure<instancing> accelerationStructure [[ buffer(BufferIndexAccelerationStructure) ]],
                                   device Resource *resources [[ buffer(BufferIndexResources) ]],
                                   device MTLAccelerationStructureInstanceDescriptor *instances [[ buffer(BufferIndexInstanceDescriptors) ]],
                                   device Light *lights [[ buffer(BufferIndexLights) ]],
                                   texture2d<unsigned int, access::read> randomTexture [[ texture(TextureIndexRandom) ]],
                                   texture2d<float, access::read> prevTex [[ texture(TextureIndexAccumulation) ]],
                                   texture2d<float, access::write> dstTex [[ texture(TextureIndexPreviousAccumulation) ]])
{
    float2 pixel = in.position.xy;
    
    pixel.y = dstTex.get_height() - pixel.y;
    
    ushort2 pos = (ushort2)in.position.xy;
    unsigned int offset = randomTexture.read(pos).x;
    
    // Add a random offset to the pixel coordinates for antialiasing.
    float2 r = float2(halton2(offset + uniforms.frameIndex, 0),
                      halton2(offset + uniforms.frameIndex, 1));
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
        
        float3 objectSpaceSurfaceNormal = interpolateVertexAttribute2(resource.normals, intersection, resource.indices);
        float3 worldSpaceSurfaceNormal = (objectToWorldSpaceTransform * float4(objectSpaceSurfaceNormal, 0)).xyz;
        worldSpaceSurfaceNormal = normalize(worldSpaceSurfaceNormal);
        float3 surfaceColor = resource.material->baseColor;
        
        // Choose a random light source to sample.
        float lightSample = halton2(offset + uniforms.frameIndex, 2 + bounce * 5 + 0);
        int lightIndex = min((int)(lightSample * uniforms.lightCount), uniforms.lightCount - 1);
        
        device Light &light = lights[lightIndex];
        
        float3 worldSpaceLightDirection;
        float lightDistance;
        float3 lightColor;
        
        if (light.type == LightTypeAreaLight) {
            
            // Choose a random point to sample on the light source.
            r = float2(halton2(offset + uniforms.frameIndex, 2 + bounce * 5 + 1),
                              halton2(offset + uniforms.frameIndex, 2 + bounce * 5 + 2));

            // Sample the lighting between the intersection point and the point on the area light.
            sampleAreaLight2(light, r, worldSpaceIntersectionPoint, worldSpaceLightDirection,
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
        r = float2(halton2(offset + uniforms.frameIndex, 2 + bounce * 5 + 3),
                   halton2(offset + uniforms.frameIndex, 2 + bounce * 5 + 4));

        float3 worldSpaceSampleDirection = sampleCosineWeightedHemisphere2(r);
        worldSpaceSampleDirection = alignHemisphereWithNormal2(worldSpaceSampleDirection, worldSpaceSurfaceNormal);

        ray.origin = worldSpaceIntersectionPoint + worldSpaceSurfaceNormal * 1e-3f;
        ray.direction = worldSpaceSampleDirection;
    }
                
    // Average this frame's sample with all of the previous frames.
    if (uniforms.frameIndex > 0) {
        float3 prevColor = prevTex.read(pos).xyz;
        prevColor *= uniforms.frameIndex;

        accumulatedColor += prevColor;
        accumulatedColor /= (uniforms.frameIndex + 1);
    }

    dstTex.write(float4(accumulatedColor, 1.0f), pos);
    
    return float4(accumulatedColor, 1.0);
}
