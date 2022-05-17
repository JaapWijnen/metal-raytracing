import MetalKit
import MetalPerformanceShaders

class SplitComputeRenderer: Renderer {
    var resetRayColorPipeline: MTLComputePipelineState!
    var resetAccumulationTexturePipeline: MTLComputePipelineState!
    var primaryRayPipeline: MTLComputePipelineState!
    var intersectionPipeline: MTLComputePipelineState!
    var shadePipeline: MTLComputePipelineState!
    var shadowIntersectionPipeline: MTLComputePipelineState!
    var accumulationPipeline: MTLComputePipelineState!
    
    var renderPipeline: MTLRenderPipelineState!
    
    var accumulationTarget: MTLTexture!
    var renderTarget: MTLTexture!
    var randomTexture: MTLTexture!
    var rayColorTexture: MTLTexture!
    
    var rayBuffer: MTLBuffer!
    var shadowRayBuffer: MTLBuffer!
    var intersectionBuffer: MTLBuffer!
    
    var lightColorBuffer: MTLBuffer!
    
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
            computeDescriptor.computeFunction = library.makeFunction(name: "primary_ray_kernel")
            computeDescriptor.label = "Primary Ray PipelineState"
            primaryRayPipeline = try device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)
            computeDescriptor.computeFunction = library.makeFunction(name: "intersection_kernel")
            computeDescriptor.label = "Intersection PipelineState"
            intersectionPipeline = try device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)
            computeDescriptor.computeFunction = try library.makeFunction(name: "shade_kernel", constantValues: shadeFunctionConstants)
            computeDescriptor.label = "Shade PipelineState"
            shadePipeline = try device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)
            computeDescriptor.computeFunction = library.makeFunction(name: "shadow_intersection_kernel")
            computeDescriptor.label = "Shadow Intersection PipelineState"
            shadowIntersectionPipeline = try device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)
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
        
        accumulationTarget = device.makeTexture(descriptor: descriptor)!
        accumulationTarget.label = "Accumulation Target"
        
        renderTarget = device.makeTexture(descriptor: descriptor)!
        renderTarget.label = "Render Target"
        
        rayColorTexture = device.makeTexture(descriptor: descriptor)!
        rayColorTexture.label = "Ray color Texture"

        // Create a texture containing a random integer value for each pixel. the sample
        // uses these values to decorrelate pixels while drawing pseudorandom numbers from the
        // Halton sequence.
        descriptor.pixelFormat = .r32Uint
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared

        randomTexture = device.makeTexture(descriptor: descriptor)
        randomTexture.label = "Random Texture"
        
        // Initialize random values.
        let numbers = Array<UInt32>.init(unsafeUninitializedCapacity: randomTexture.width * randomTexture.height) { buffer, initializedCount in
            for i in 0..<randomTexture.width * randomTexture.height {
                buffer[i] = arc4random() % (1024 * 1024)
            }
            initializedCount = randomTexture.width * randomTexture.height
        }
        
        numbers.withUnsafeBufferPointer { bufferPointer in
            randomTexture.replace(
                region: .init(
                    origin: .init(x: 0, y: 0, z: 0),
                    size: .init(width: randomTexture.width, height: randomTexture.height, depth: 1)
                ),
                mipmapLevel: 0,
                withBytes: bufferPointer.baseAddress!,
                bytesPerRow: MemoryLayout<UInt32>.stride * randomTexture.width
            )
        }
    }
    
    override func resizeBuffers(size: CGSize) {
        let rayCount = Int(size.width * size.height)
        let rayStride = MemoryLayout<Ray>.stride
        let intersectionStride = MemoryLayout<Intersection>.stride
        
        rayBuffer = device.makeBuffer(length: rayCount * rayStride, options: .storageModePrivate)
        rayBuffer.label = "Ray Buffer"
        shadowRayBuffer = device.makeBuffer(length: rayCount * rayStride, options: .storageModePrivate)
        shadowRayBuffer.label = "Shadow Ray Buffer"
        
        intersectionBuffer = device.makeBuffer(length: rayCount * intersectionStride, options: .storageModePrivate)
        intersectionBuffer.label = "Intersection Buffer"
        
        lightColorBuffer = device.makeBuffer(length: rayCount * MemoryLayout<SIMD4<Float>>.stride, options: .storageModePrivate)
        lightColorBuffer.label = "Light Color Buffer"
    }
    
    override func draw(in view: MTKView) {
        semaphore.wait()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        commandBuffer.addCompletedHandler { _ in self.semaphore.signal() }
        
        let size = view.drawableSize
        update(size: size)
        
        let width = Int(size.width)
        let height = Int(size.height)
        
        // TODO: check logic
        let threadsPerGroup = MTLSize(width: 8, height: 8, depth: 1)
        let threadGroups = MTLSize(width: (width + threadsPerGroup.width - 1) / threadsPerGroup.width, height: (height + threadsPerGroup.height - 1) / threadsPerGroup.height, depth: 1)
        
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Reset Raycolor pass"
        computeEncoder.setComputePipelineState(resetRayColorPipeline)
        //computeEncoder.setBuffer(rayColorBuffer, offset: 0, index: BufferIndex.rayColors.rawValue)
        computeEncoder.setTexture(rayColorTexture, index: TextureIndex.rayColor.rawValue)
        computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        computeEncoder.endEncoding()
        
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Reset Accumulation texture pass"
        computeEncoder.setComputePipelineState(resetAccumulationTexturePipeline)
        computeEncoder.setTexture(accumulationTarget, index: TextureIndex.accumulationTarget.rawValue)
        computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        computeEncoder.endEncoding()
        
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Primary ray pass"
        computeEncoder.setComputePipelineState(primaryRayPipeline)
        computeEncoder.setBuffer(rayBuffer, offset: 0, index: BufferIndex.rays.rawValue)
        computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset)
        computeEncoder.setTexture(randomTexture, index: TextureIndex.random.rawValue)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        computeEncoder.endEncoding()
        
        for bounce in 0..<3 {
            guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
            computeEncoder.label = "Intersection pass"
            computeEncoder.setComputePipelineState(intersectionPipeline)
            computeEncoder.setBuffer(rayBuffer, offset: 0, index: BufferIndex.rays.rawValue)
            computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset)
            computeEncoder.setBuffer(intersectionBuffer, offset: 0, index: BufferIndex.intersections.rawValue)
            
            computeEncoder.setAccelerationStructure(scene.instancedAccelarationStructure, bufferIndex: BufferIndex.accelerationStructure.rawValue)
            for accelerationStructure in scene.primitiveAccelerationStructures {
                computeEncoder.useResource(accelerationStructure, usage: .read)
            }
            
            computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
            computeEncoder.endEncoding()
            
            guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
            computeEncoder.label = "Shading pass"
            computeEncoder.setComputePipelineState(shadePipeline)
            var bounce = bounce
            computeEncoder.setBytes(&bounce, length: MemoryLayout<Int>.stride, index: BufferIndex.bounce.rawValue)
            computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset)
            computeEncoder.setBuffer(intersectionBuffer, offset: 0, index: BufferIndex.intersections.rawValue)
            computeEncoder.setBuffer(scene.instanceDescriptorBuffer, offset: 0, index: BufferIndex.instanceDescriptors.rawValue)
            computeEncoder.setBuffer(scene.resourcesBuffer, offset: 0, index: BufferIndex.resources.rawValue)
            computeEncoder.setBuffer(scene.lightBuffer)
            computeEncoder.setBuffer(lightColorBuffer, offset: 0, index: BufferIndex.lightColors.rawValue)
            computeEncoder.setTexture(rayColorTexture, index: TextureIndex.rayColor.rawValue)
            computeEncoder.setBuffer(rayBuffer, offset: 0, index: BufferIndex.rays.rawValue)
            computeEncoder.setBuffer(shadowRayBuffer, offset: 0, index: BufferIndex.shadowRays.rawValue)
            computeEncoder.setTexture(randomTexture, index: TextureIndex.random.rawValue)
            
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
            computeEncoder.label = "Shadow intersection pass"
            computeEncoder.setComputePipelineState(shadowIntersectionPipeline)
            computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset)
            computeEncoder.setBuffer(shadowRayBuffer, offset: 0, index: BufferIndex.shadowRays.rawValue)
            computeEncoder.setBuffer(lightColorBuffer, offset: 0, index: BufferIndex.lightColors.rawValue)
            computeEncoder.setTexture(rayColorTexture, index: TextureIndex.rayColor.rawValue)
            computeEncoder.setTexture(accumulationTarget, index: TextureIndex.accumulationTarget.rawValue)
            
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
        computeEncoder.setTexture(accumulationTarget, index: TextureIndex.accumulationTarget.rawValue)
        computeEncoder.setTexture(renderTarget, index: TextureIndex.renderTarget.rawValue)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        computeEncoder.endEncoding()
        
        guard let descriptor = view.currentRenderPassDescriptor, let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }
        renderEncoder.label = "Accumulation render pass"
        renderEncoder.setRenderPipelineState(renderPipeline)
        renderEncoder.setFragmentBuffer(uniformBuffer, offset: uniformBufferOffset)
        renderEncoder.setFragmentTexture(renderTarget, index: TextureIndex.renderTarget.rawValue)
        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        renderEncoder.endEncoding()
        
        guard let drawable = view.currentDrawable else { return }
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
