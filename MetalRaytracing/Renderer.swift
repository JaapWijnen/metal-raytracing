//
//  Renderer.swift
//  MetalRaytracing
//
//  Created by Jaap Wijnen on 21/11/2021.
//

import Metal
import MetalKit
import simd

class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary
    
    let semaphore: DispatchSemaphore
    
    let scene: Scene
    
    weak var viewController: GameViewController?
    
    func createPipelineStates(for mtkView: MTKView) { }
    func resizeTextures(size: CGSize) { }
    func resizeBuffers(size: CGSize) { }
    
    let maxFramesInFlight = 3
    
    var uniformBufferIndex = 0
    var uniformBufferOffset: Int {
        uniformBufferIndex * MemoryLayout<Uniforms>.stride
    }
    var uniformBuffer: Buffer<Uniforms>!
    
    var frameIndex: UInt32 = 0
    
    init?(metalView: MTKView, viewController: GameViewController) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("GPU not available")
        }
        
        self.viewController = viewController
        
        let size = metalView.frame.size
        
        metalView.device = device
        metalView.colorPixelFormat = .rgba16Float // .rgba16Float
        metalView.sampleCount = 1
        metalView.drawableSize = size
        
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        self.library = device.makeDefaultLibrary()!
        
        self.scene = DragonScene(size: size, device: device)
        
        self.semaphore = DispatchSemaphore.init(value: maxFramesInFlight)
        
        super.init()
        metalView.delegate = self
        mtkView(metalView, drawableSizeWillChange: size)
        
        uniformBuffer = device.makeBuffer(
            count: maxFramesInFlight,
            index: .uniforms,
            label: "Uniform Buffer",
            options: .storageModeShared
        )!
        
        scene.createResourceBuffer(on: device)
        createPipelineStates(for: metalView)
        scene.createAccelerationStructures(on: commandQueue)
    }
    
    func updateUniforms(size: CGSize) {
        scene.updateUniforms(size: size)
        
        let pointer = uniformBuffer.buffer.contents().advanced(by: uniformBufferOffset)
        let uniforms = pointer.bindMemory(to: Uniforms.self, capacity: 1)
        
        uniforms.pointee.camera = scene.camera
        uniforms.pointee.lightCount = Int32(scene.lights.count)
        uniforms.pointee.width = Int32(size.width)
        uniforms.pointee.height = Int32(size.height)
        uniforms.pointee.blocksWide = ((uniforms.pointee.width) + 15) / 16
        uniforms.pointee.frameIndex = frameIndex
        frameIndex += 1
    }
    
    func update(size: CGSize) {
        viewController?.switchRenderMode()
        updateUniforms(size: size)
        uniformBufferIndex = (uniformBufferIndex + 1) % maxFramesInFlight
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        resizeTextures(size: size)
        resizeBuffers(size: size)
        frameIndex = 0
    }
    
    func draw(in view: MTKView) {
        let size = view.drawableSize
        update(size: size)
    }
}
