//
//  GameViewController.swift
//  MetalRaytracing
//
//  Created by Jaap Wijnen on 21/11/2021.
//

import Cocoa
import MetalKit

class GameViewController: NSViewController {
    var renderer: Renderer!

    override func viewDidLoad() {
        super.viewDidLoad()

        guard let mtkView = self.view as? MTKView else {
            print("View attached to GameViewController is not an MTKView")
            return
        }
        
        // Select the device to render with.  We choose the default device
        guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return
        }

        mtkView.device = defaultDevice

        guard let newRenderer = RayBinningRenderer(metalView: mtkView, viewController: self) else {
            print("Renderer cannot be initialized")
            return
        }

        renderer = newRenderer

        renderer.mtkView(mtkView, drawableSizeWillChange: mtkView.drawableSize)

        mtkView.delegate = renderer
    }
    
    func set(renderer: Renderer) {
        self.renderer = renderer
        guard let mtkView = self.view as? MTKView else { return }
        renderer.mtkView(mtkView, drawableSizeWillChange: mtkView.drawableSize)
        mtkView.delegate = renderer
    }
    
    func switchRenderMode() {
        guard let mtkView = self.view as? MTKView else { return }
        if InputController.shared.keysPressed.contains(.one) {
            guard let renderer = Renderer(metalView: mtkView, viewController: self) else { return }
            self.set(renderer: renderer)
            print("switched to base renderer")
        }
        if InputController.shared.keysPressed.contains(.two) {
            guard let renderer = SingleComputeRenderer(metalView: mtkView, viewController: self) else { return }
            self.set(renderer: renderer)
            print("switched to single compute renderer")
        }
        if InputController.shared.keysPressed.contains(.three) {
            guard let renderer = SingleFragmentRenderer(metalView: mtkView, viewController: self) else { return }
            self.set(renderer: renderer)
            print("switched to single fragment renderer")
        }
        if InputController.shared.keysPressed.contains(.four) {
            guard let renderer = SplitComputeRenderer(metalView: mtkView, viewController: self) else { return }
            self.set(renderer: renderer)
            print("switched to split compute renderer")
        }
        if InputController.shared.keysPressed.contains(.five) {
            guard let renderer = SplitDataRenderer(metalView: mtkView, viewController: self) else { return }
            self.set(renderer: renderer)
            print("switched to split data renderer")
        }
        if InputController.shared.keysPressed.contains(.six) {
            guard let renderer = RayBinningRenderer(metalView: mtkView, viewController: self) else { return }
            self.set(renderer: renderer)
            print("switched to ray binning renderer")
        }
    }
}
