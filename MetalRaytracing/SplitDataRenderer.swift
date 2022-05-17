import MetalKit
import MetalPerformanceShaders

class SplitDataRenderer: Renderer {
    var resetRayColorPipeline: MTLComputePipelineState!
    var resetAccumulationTexturePipeline: MTLComputePipelineState!
    var primaryRayPipeline: MTLComputePipelineState!
    var intersectionPipeline: MTLComputePipelineState!
    var shadePipeline: MTLComputePipelineState!
    var shade2Pipeline: MTLComputePipelineState!
    var shadowIntersectionPipeline: MTLComputePipelineState!
    var updateRaysPipeline: MTLComputePipelineState!
    var accumulationPipeline: MTLComputePipelineState!
    
    var renderPipeline: MTLRenderPipelineState!
    
    var accumulationTarget: Texture!
    var renderTarget: Texture!
    var randomTexture: Texture!
    var surfaceColorTexture: Texture!
    var worldSpaceSurfaceNormalTexture: Texture!
    
    var rayOriginTexture: Texture!
    var rayDirectionTexture: Texture!
    var rayMaxDistanceBuffer: Buffer<Float>!

    var rayColorTexture: Texture!
    
    var shadowRayOriginTexture: Texture!
    var shadowRayDirectionTexture: Texture!
    var shadowRayMaxDistanceBuffer: Buffer<Float>!
    
    var intersectionDistanceBuffer: Buffer<Float>!
    var intersectionInstanceIDBuffer: Buffer<UInt8>!
    var intersectionGeometryIDBuffer: Buffer<UInt8>!
    var intersectionPrimitiveIDBuffer: Buffer<UInt32>!
    var intersectionCoordinatesBuffer: Buffer<Float>!
    var intersectionWorldSpaceIntersectionPointBuffer: Buffer<SIMD3<Float>>!
    
    var lightColorBuffer: Buffer<SIMD4<Float>>!
    
    override init?(metalView: MTKView, viewController: GameViewController) {
        super.init(metalView: metalView, viewController: viewController)
        self.resizeBuffers(size: metalView.drawableSize)
    }
    
    override func createPipelineStates(for mtkView: MTKView) {
        let vertexFunction = library.makeFunction(name: "vertex_split_rt")
        let fragmentFunction = library.makeFunction(name: "fragment_split_rt")
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.sampleCount = mtkView.sampleCount
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat
        
        let computeDescriptor = MTLComputePipelineDescriptor()
        computeDescriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
        
        let shadeFunctionConstants = MTLFunctionConstantValues()
        var resourceStride = Int32(scene.resourceStride)
        shadeFunctionConstants.setConstantValue(&resourceStride, type: .int, index: 0)
        var maxSubmeshes = Int32(scene.maxSubmeshes)
        shadeFunctionConstants.setConstantValue(&maxSubmeshes, type: .int, index: 1)
        
        do {
            computeDescriptor.computeFunction = library.makeFunction(name: "reset_ray_color_kernel")
            computeDescriptor.label = "Reset ray color PipelineState"
            resetRayColorPipeline = try device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)
            computeDescriptor.computeFunction = library.makeFunction(name: "reset_accumulation_texture_kernel")
            computeDescriptor.label = "Reset accumulation texture PipelineState"
            resetAccumulationTexturePipeline = try device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)
            computeDescriptor.computeFunction = library.makeFunction(name: "sd_primary_ray_kernel")
            computeDescriptor.label = "Primary Ray PipelineState"
            primaryRayPipeline = try device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)
            computeDescriptor.computeFunction = library.makeFunction(name: "sd_intersection_kernel")
            computeDescriptor.label = "Intersection PipelineState"
            intersectionPipeline = try device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)
            computeDescriptor.computeFunction = try library.makeFunction(name: "sd_shade_kernel", constantValues: shadeFunctionConstants)
            computeDescriptor.label = "Shade PipelineState"
            shadePipeline = try device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)
            computeDescriptor.computeFunction = try library.makeFunction(name: "sd_shade2_kernel", constantValues: shadeFunctionConstants)
            computeDescriptor.label = "Shade 2 PipelineState"
            shade2Pipeline = try device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)
            computeDescriptor.computeFunction = library.makeFunction(name: "sd_shadow_intersection_kernel")
            computeDescriptor.label = "Shadow Intersection PipelineState"
            shadowIntersectionPipeline = try device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)
            computeDescriptor.computeFunction = library.makeFunction(name: "sd_update_rays_kernel")
            computeDescriptor.label = "Update Rays PipelineState"
            updateRaysPipeline = try device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)
            computeDescriptor.computeFunction = library.makeFunction(name: "accumulation_kernel")
            computeDescriptor.label = "Accumulation PipelineState"
            accumulationPipeline = try device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)
            
            renderPipeline = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print(error.localizedDescription)
        }
    }
    
    override func resizeTextures(size: CGSize) {
        let descriptor = MTLTextureDescriptor()
        descriptor.pixelFormat = .rgba32Float
        descriptor.textureType = .type2D
        descriptor.width = Int(size.width)
        descriptor.height = Int(size.height)

        // Stored in private memory because only the GPU will read or write this texture.
        descriptor.storageMode = .private
        descriptor.usage = [.shaderRead, .shaderWrite]
        
        accumulationTarget = device.makeTexture(descriptor: descriptor, index: .accumulationTarget, label: "Accumulation Target")!
        renderTarget = device.makeTexture(descriptor: descriptor, index: .renderTarget, label: "Render Target")!
        
        rayColorTexture = device.makeTexture(descriptor: descriptor, index: .rayColor, label: "Ray color Texture")!
        
        surfaceColorTexture = device.makeTexture(descriptor: descriptor, index: .surfaceColor, label: "Surface color Texture")!
        
        worldSpaceSurfaceNormalTexture = device.makeTexture(descriptor: descriptor, index: .worldSpaceSurfaceNormal, label: "World Space Surface Normal Texture")!
        
        rayOriginTexture = device.makeTexture(descriptor: descriptor, index: .rayOrigins, label: "Ray Origin Texture")!
        rayDirectionTexture = device.makeTexture(descriptor: descriptor, index: .rayDirections, label: "Ray Direction Texture")!
        
        shadowRayOriginTexture = device.makeTexture(descriptor: descriptor, index: .shadowRayOrigins, label: "Shadow Ray Origin Texture")!
        shadowRayDirectionTexture = device.makeTexture(descriptor: descriptor, index: .shadowRayDirections, label: "Shadow Ray Direction Texture")!

        // Create a texture containing a random integer value for each pixel. the sample
        // uses these values to decorrelate pixels while drawing pseudorandom numbers from the
        // Halton sequence.
        descriptor.pixelFormat = .r32Uint
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared

        randomTexture = device.makeTexture(descriptor: descriptor, index: .random, label: "Random Texture")
        
        // Initialize random values.
        let numbers = Array<UInt32>.init(unsafeUninitializedCapacity: randomTexture.texture.width * randomTexture.texture.height) { buffer, initializedCount in
            for i in 0..<randomTexture.texture.width * randomTexture.texture.height {
                buffer[i] = arc4random() % (1024 * 1024)
            }
            initializedCount = randomTexture.texture.width * randomTexture.texture.height
        }
        
        numbers.withUnsafeBufferPointer { bufferPointer in
            randomTexture.texture.replace(
                region: .init(
                    origin: .init(x: 0, y: 0, z: 0),
                    size: .init(width: randomTexture.texture.width, height: randomTexture.texture.height, depth: 1)
                ),
                mipmapLevel: 0,
                withBytes: bufferPointer.baseAddress!,
                bytesPerRow: MemoryLayout<UInt32>.stride * randomTexture.texture.width
            )
        }
    }
    
    override func resizeBuffers(size: CGSize) {
        let rayCount = Int(size.width * size.height)
        
        rayMaxDistanceBuffer = device.makeBuffer(count: rayCount, index: .rayMaxDistances, label: "Ray Max Distance Buffer", options: .storageModePrivate)
        shadowRayMaxDistanceBuffer = device.makeBuffer(count: rayCount, index: .shadowRayMaxDistances, label: "Shadow Ray Max Distance Buffer", options: .storageModePrivate)
        
//        rayBuffer = device.makeBuffer(length: rayCount * rayStride, options: .storageModePrivate)
//        rayBuffer.label = "Ray Buffer"
//        shadowRayBuffer = device.makeBuffer(length: rayCount * rayStride, options: .storageModePrivate)
//        shadowRayBuffer.label = "Shadow Ray Buffer"
        
        intersectionDistanceBuffer = device.makeBuffer(count: rayCount, index: .intersectionDistances, label: "Intersection Distance Buffer", options: .storageModePrivate)
        intersectionInstanceIDBuffer = device.makeBuffer(count: rayCount, index: .intersectionInstanceIDs, label: "Intersection InstanceID Buffer", options: .storageModePrivate)
        intersectionGeometryIDBuffer = device.makeBuffer(count: rayCount, index: .intersectionGeometryIDs, label: "Intersection GeometryID Buffer", options: .storageModePrivate)
        intersectionPrimitiveIDBuffer = device.makeBuffer(count: rayCount, index: .intersectionPrimitiveIDs, label: "Intersection PrimitiveID Buffer", options: .storageModePrivate)
        intersectionCoordinatesBuffer = device.makeBuffer(count: rayCount, index: .intersectionTriangleCoordinates, label: "Intersection Coordinates Buffer", options: .storageModePrivate)
        intersectionWorldSpaceIntersectionPointBuffer = device.makeBuffer(count: rayCount, index: .intersectionWorldSpaceIntersectionPoints, label: "Intersection World Space Intersection Point Buffer", options: .storageModePrivate)
        lightColorBuffer = device.makeBuffer(count: rayCount, index: .lightColors, label: "Light Color Buffer", options: .storageModePrivate)
    }
    
    override func draw(in view: MTKView) {
        semaphore.wait()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        commandBuffer.addCompletedHandler { _ in self.semaphore.signal() }
        
        let size = view.drawableSize
        update(size: size)
        
        let width = Int(size.width)
        let height = Int(size.height)
        
        let pipelines: [MTLComputePipelineState] = [resetRayColorPipeline, resetAccumulationTexturePipeline, primaryRayPipeline, intersectionPipeline, shadePipeline, shade2Pipeline, shadowIntersectionPipeline, updateRaysPipeline, accumulationPipeline]
        
        let _ = pipelines.map { p in
            print(p.label, p.maxTotalThreadsPerThreadgroup)
        }
        
        // TODO: check logic
        let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(width: (width + threadsPerGroup.width - 1) / threadsPerGroup.width, height: (height + threadsPerGroup.height - 1) / threadsPerGroup.height, depth: 1)
        
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Reset Raycolor pass"
        computeEncoder.setComputePipelineState(resetRayColorPipeline)
        //computeEncoder.setBuffer(rayColorBuffer, offset: 0, index: BufferIndex.rayColors.rawValue)
        computeEncoder.setTexture(rayColorTexture)
        computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        computeEncoder.endEncoding()
        
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Reset Accumulation texture pass"
        computeEncoder.setComputePipelineState(resetAccumulationTexturePipeline)
        computeEncoder.setTexture(accumulationTarget)
        computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        computeEncoder.endEncoding()
        
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Primary ray pass"
        computeEncoder.setComputePipelineState(primaryRayPipeline)
        computeEncoder.setTexture(rayOriginTexture)
        computeEncoder.setTexture(rayDirectionTexture)
        computeEncoder.setBuffer(rayMaxDistanceBuffer)
        computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset)
        computeEncoder.setTexture(randomTexture)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        computeEncoder.endEncoding()
        
        let maxBounces = 3
        
        for bounce in 0..<maxBounces {
            guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
            computeEncoder.label = "Intersection pass"
            computeEncoder.setComputePipelineState(intersectionPipeline)
            computeEncoder.setTexture(rayOriginTexture)
            computeEncoder.setTexture(rayDirectionTexture)
            computeEncoder.setBuffer(rayMaxDistanceBuffer)
            computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset)
            
            computeEncoder.setBuffer(intersectionDistanceBuffer)
            computeEncoder.setBuffer(intersectionInstanceIDBuffer)
            computeEncoder.setBuffer(intersectionGeometryIDBuffer)
            computeEncoder.setBuffer(intersectionPrimitiveIDBuffer)
            computeEncoder.setBuffer(intersectionCoordinatesBuffer)
            computeEncoder.setBuffer(intersectionWorldSpaceIntersectionPointBuffer)
            
            computeEncoder.setAccelerationStructure(scene.instancedAccelarationStructure, bufferIndex: BufferIndex.accelerationStructure.rawValue)
            for accelerationStructure in scene.primitiveAccelerationStructures {
                computeEncoder.useResource(accelerationStructure, usage: .read)
            }
            
            computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
            computeEncoder.endEncoding()
            
            guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
            computeEncoder.label = "Shading pass"
            computeEncoder.setComputePipelineState(shadePipeline)
            
            computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset)
            computeEncoder.setBuffer(scene.instanceDescriptorBuffer, offset: 0, index: BufferIndex.instanceDescriptors.rawValue)
            computeEncoder.setBuffer(scene.resourcesBuffer, offset: 0, index: BufferIndex.resources.rawValue)
            

            computeEncoder.setBuffer(rayMaxDistanceBuffer)
            computeEncoder.setBuffer(shadowRayMaxDistanceBuffer)
            
            computeEncoder.setBuffer(intersectionDistanceBuffer)
            computeEncoder.setBuffer(intersectionInstanceIDBuffer)
            computeEncoder.setBuffer(intersectionGeometryIDBuffer)
            computeEncoder.setBuffer(intersectionPrimitiveIDBuffer)
            computeEncoder.setBuffer(intersectionCoordinatesBuffer)
            
            computeEncoder.setTexture(randomTexture)
            computeEncoder.setTexture(surfaceColorTexture)
            computeEncoder.setTexture(worldSpaceSurfaceNormalTexture)
            
            for model in scene.models {
                for mesh in model.meshes {
                    for resource in mesh.resources {
                        computeEncoder.useResource(resource, usage: .read)
                    }
                }
            }
            
            computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
            computeEncoder.endEncoding()
            
            guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
            computeEncoder.label = "Shading pass 2"
            computeEncoder.setComputePipelineState(shade2Pipeline)
            var bounce = bounce
            computeEncoder.setBytes(&bounce, length: MemoryLayout<Int>.stride, index: BufferIndex.bounce.rawValue)
            computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset)
            computeEncoder.setBuffer(scene.lightBuffer)
            computeEncoder.setBuffer(lightColorBuffer)
            computeEncoder.setTexture(rayColorTexture)
            
            computeEncoder.setTexture(rayOriginTexture)
            computeEncoder.setTexture(rayDirectionTexture)
            computeEncoder.setBuffer(rayMaxDistanceBuffer)
            
            computeEncoder.setTexture(shadowRayOriginTexture)
            computeEncoder.setTexture(shadowRayDirectionTexture)
            computeEncoder.setBuffer(shadowRayMaxDistanceBuffer)
            
            computeEncoder.setBuffer(intersectionWorldSpaceIntersectionPointBuffer)
            
            computeEncoder.setTexture(randomTexture)
            computeEncoder.setTexture(surfaceColorTexture)
            computeEncoder.setTexture(worldSpaceSurfaceNormalTexture)
            
            computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
            computeEncoder.endEncoding()
            
            if bounce < maxBounces - 1 {
                guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
                computeEncoder.label = "Update Rays pass"
                computeEncoder.setComputePipelineState(updateRaysPipeline)
                computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset)
                var bounce = bounce
                computeEncoder.setBytes(&bounce, length: MemoryLayout<Int>.stride, index: BufferIndex.bounce.rawValue)
                computeEncoder.setBuffer(intersectionWorldSpaceIntersectionPointBuffer)
                computeEncoder.setTexture(randomTexture)
                computeEncoder.setTexture(worldSpaceSurfaceNormalTexture)
                computeEncoder.setTexture(rayOriginTexture)
                computeEncoder.setTexture(rayDirectionTexture)
                computeEncoder.setBuffer(rayMaxDistanceBuffer)
                
                computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
                computeEncoder.endEncoding()
            }
            
            guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
            computeEncoder.label = "Shadow intersection pass"
            computeEncoder.setComputePipelineState(shadowIntersectionPipeline)
            computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset)
            
            computeEncoder.setTexture(shadowRayOriginTexture)
            computeEncoder.setTexture(shadowRayDirectionTexture)
            computeEncoder.setBuffer(shadowRayMaxDistanceBuffer)
            
            computeEncoder.setBuffer(lightColorBuffer)
            computeEncoder.setTexture(rayColorTexture)
            computeEncoder.setTexture(accumulationTarget)
            
            computeEncoder.setAccelerationStructure(scene.instancedAccelarationStructure, bufferIndex: BufferIndex.accelerationStructure.rawValue)

            for accelerationStructure in scene.primitiveAccelerationStructures {
                computeEncoder.useResource(accelerationStructure, usage: .read)
            }
            
            computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
            computeEncoder.endEncoding()
        }
        
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Accumulation pass"
        computeEncoder.setComputePipelineState(accumulationPipeline)
        computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset)
        computeEncoder.setTexture(accumulationTarget)
        computeEncoder.setTexture(renderTarget)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        computeEncoder.endEncoding()
        
        guard let descriptor = view.currentRenderPassDescriptor, let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }
        renderEncoder.label = "Accumulation render pass"
        renderEncoder.setRenderPipelineState(renderPipeline)
        renderEncoder.setFragmentBuffer(uniformBuffer, offset: uniformBufferOffset)
        renderEncoder.setFragmentTexture(renderTarget)
        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        renderEncoder.endEncoding()
        
        guard let drawable = view.currentDrawable else { return }
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
