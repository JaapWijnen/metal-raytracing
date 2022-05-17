import MetalKit

class SingleFragmentRenderer: Renderer {
    var fragmentRaytracingPipelineState: MTLRenderPipelineState!
    
    var accumulationTargets: [MTLTexture] = []
    var randomTexture: MTLTexture!
    
    override func createPipelineStates(for metalView: MTKView) {
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.sampleCount = metalView.sampleCount
        pipelineDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
        
        let functionConstants = MTLFunctionConstantValues()
        var resourceStride = Int32(scene.resourceStride)
        functionConstants.setConstantValue(&resourceStride, type: .int, index: 0)
        var maxSubmeshes = Int32(scene.maxSubmeshes)
        functionConstants.setConstantValue(&maxSubmeshes, type: .int, index: 1)
        
        let rtVertexFunction = library.makeFunction(name: "raytracingVertex")
        
        pipelineDescriptor.vertexFunction = rtVertexFunction
        pipelineDescriptor.label = "Fragment Raytracing PipelineState"
        
        do {
            let rtFragmentFunction = try library.makeFunction(name: "raytracingFragment", constantValues: functionConstants)
            pipelineDescriptor.fragmentFunction = rtFragmentFunction
            fragmentRaytracingPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
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
        commandBuffer.addCompletedHandler { thing in self.semaphore.signal() }
        
        let size = view.drawableSize
        update(size: size)
        
        guard let descriptor = view.currentRenderPassDescriptor else { return }
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }
        renderEncoder.label = "Fragment Raytracing Pass"
        renderEncoder.setRenderPipelineState(fragmentRaytracingPipelineState)
                
        renderEncoder.setFragmentBuffer(uniformBuffer, offset: uniformBufferOffset)
        renderEncoder.setFragmentBuffer(scene.resourcesBuffer,          offset: 0,                   index: BufferIndex.resources.rawValue)
        renderEncoder.setFragmentBuffer(scene.instanceDescriptorBuffer, offset: 0,                   index: BufferIndex.instanceDescriptors.rawValue)
        renderEncoder.setFragmentBuffer(scene.lightBuffer)
    
        renderEncoder.setFragmentAccelerationStructure(scene.instancedAccelarationStructure, bufferIndex: BufferIndex.accelerationStructure.rawValue)
    
        renderEncoder.setFragmentTexture(randomTexture,          index: TextureIndex.random.rawValue)
        renderEncoder.setFragmentTexture(accumulationTargets[0], index: TextureIndex.accumulationTarget.rawValue)
        renderEncoder.setFragmentTexture(accumulationTargets[1], index: TextureIndex.previousAccumulation.rawValue)
        
        for model in scene.models {
            for mesh in model.meshes {
                for resource in mesh.resources {
                    renderEncoder.useResource(resource, usage: .read)
                }
            }
        }
        
        for accelerationStructure in scene.primitiveAccelerationStructures {
            renderEncoder.useResource(accelerationStructure, usage: .read)
        }
        
        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        renderEncoder.endEncoding()
        
        let tmp = accumulationTargets[0]
        accumulationTargets[0] = accumulationTargets[1]
        accumulationTargets[1] = tmp
        
        guard let drawable = view.currentDrawable else { return }
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
