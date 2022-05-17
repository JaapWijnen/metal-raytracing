import MetalKit

class SingleComputeRenderer: Renderer {
    var raytracingPipelineState: MTLComputePipelineState!
    var renderPipelineState: MTLRenderPipelineState!
    
    var accumulationTargets: [MTLTexture] = []
    var randomTexture: MTLTexture!
    
    override func createPipelineStates(for metalView: MTKView) {
        let vertexFunction = library.makeFunction(name: "vertexShader")
        let fragmentFunction = library.makeFunction(name: "fragmentShader")
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.sampleCount = metalView.sampleCount
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
        pipelineDescriptor.label = "Render PipelineState"
        
        do {
            renderPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print(error.localizedDescription)
        }
        
        let functionConstants = MTLFunctionConstantValues()
        var resourceStride = Int32(scene.resourceStride)
        functionConstants.setConstantValue(&resourceStride, type: .int, index: 0)
        var maxSubmeshes = Int32(scene.maxSubmeshes)
        functionConstants.setConstantValue(&maxSubmeshes, type: .int, index: 1)
        
        let computeDescriptor = MTLComputePipelineDescriptor()
        computeDescriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
        computeDescriptor.label = "Raytracing PipelineState"
        
        do {
            computeDescriptor.computeFunction = try library.makeFunction(name: "raytracingKernel", constantValues: functionConstants)
            raytracingPipelineState = try device.makeComputePipelineState(descriptor: computeDescriptor, options: [], reflection: nil)
        } catch {
            print(error.localizedDescription)
        }
        
        print(raytracingPipelineState.threadExecutionWidth,
        raytracingPipelineState.maxTotalThreadsPerThreadgroup,
        raytracingPipelineState.staticThreadgroupMemoryLength)
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
        
        accumulationTargets = [device.makeTexture(descriptor: descriptor)!, device.makeTexture(descriptor: descriptor)!]
        accumulationTargets[0].label = "Accumulation Texture 1"
        accumulationTargets[1].label = "Accumulation Texture 2"

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
    
    override func draw(in view: MTKView) {
        semaphore.wait()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        commandBuffer.addCompletedHandler { _ in self.semaphore.signal() }
        
        let size = view.drawableSize
        update(size: size)
        
        
        let width = Int(size.width)
        let height = Int(size.height)
        // process rays in 8x8 tiles
        let threadsPerGroup = MTLSize(width: 8, height: 8, depth: 1)
        let threadGroups = MTLSize(
            width: (width + threadsPerGroup.width - 1) / threadsPerGroup.width,
            height: (height + threadsPerGroup.height - 1) / threadsPerGroup.height,
            depth: 1
        )
        
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Raytracing Pass"
        computeEncoder.setComputePipelineState(raytracingPipelineState)
    
        computeEncoder.setBuffer(uniformBuffer, offset: uniformBufferOffset)
        computeEncoder.setBuffer(scene.resourcesBuffer,          offset: 0,                   index: BufferIndex.resources.rawValue)
        computeEncoder.setBuffer(scene.instanceDescriptorBuffer, offset: 0,                   index: BufferIndex.instanceDescriptors.rawValue)
        computeEncoder.setBuffer(scene.lightBuffer)
    
        computeEncoder.setAccelerationStructure(scene.instancedAccelarationStructure, bufferIndex: BufferIndex.accelerationStructure.rawValue)
    
        computeEncoder.setTexture(randomTexture,          index: TextureIndex.random.rawValue)
        computeEncoder.setTexture(accumulationTargets[0], index: TextureIndex.accumulationTarget.rawValue)
        computeEncoder.setTexture(accumulationTargets[1], index: TextureIndex.previousAccumulation.rawValue)
        
        for model in scene.models {
            for mesh in model.meshes {
                for resource in mesh.resources {
                    computeEncoder.useResource(resource, usage: .read)
                }
            }
        }
        
        for accelerationStructure in scene.primitiveAccelerationStructures {
            computeEncoder.useResource(accelerationStructure, usage: .read)
        }
        
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        computeEncoder.endEncoding()
        
        let tmp = accumulationTargets[0]
        accumulationTargets[0] = accumulationTargets[1]
        accumulationTargets[1] = tmp
        
        guard let descriptor = view.currentRenderPassDescriptor,
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(
                descriptor: descriptor) else {
                    return
                }
        renderEncoder.setRenderPipelineState(renderPipelineState)
        
        // MARK: draw call
        renderEncoder.setFragmentTexture(accumulationTargets[0], index: 0)
        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        renderEncoder.endEncoding()
        
        guard let drawable = view.currentDrawable else { return }
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
