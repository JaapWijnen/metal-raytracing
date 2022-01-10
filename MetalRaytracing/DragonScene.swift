//
//  DragonScene.swift
//  MetalRaytracing
//
//  Created by Jaap Wijnen on 21/11/2021.
//

import MetalKit

class DragonScene: Scene {
    override init(size: CGSize, device: MTLDevice) {
        super.init(size: size, device: device)
        
        self.models = [
            Model(name: "train", position: [-0.3, 0, 0.4], scale: 0.5, on: device),
            Model(name: "dragon", position: [0.3, 0.38, 2.5], rotation: [0, .pi / 2 * 1.2, 0], scale: 1.2, on: device),
            Model(name: "treefir", position: [0.5, 0, -0.2], scale: 0.7, on: device),
            Model(name: "plane", position: [0, 0, 0], scale: 10, on: device),
            Model(name: "sphere", position: [-1.9, 0.0, 0.3], scale: 1, on: device),
            Model(name: "sphere", position: [2.9, 0.0, -0.5], scale: 2, on: device),
            Model(name: "plane-back", position: [0, 0, -1.5], scale: 10, on: device)
        ]
            
//        self.geometries = [
//            TriangleGeometry(name: "dragon", position: [-0.3, 0.38, 0.4], rotation: [0, .pi / 2 * 1.2, 0], scale: 1.2, on: device)!,
//            TriangleGeometry(name: "train", position: [-0.3, 0, -1], rotation: [0, 0, 0], scale: 0.5, on: device)!,
//            TriangleGeometry(name: "treefir", position: [0.5, 0, -0.2], rotation: [0, 0, 0], scale: 0.7, on: device)!,
//            TriangleGeometry(name: "plane", position: [0, 0, 0], rotation: [0, 0, 0], scale: 10, on: device)!,
//            TriangleGeometry(name: "sphere", position: [-1.9, 0.0, 0.3], rotation: [0, 0, 0], scale: 1, on: device)!,
//            TriangleGeometry(name: "sphere", position: [2.9, 0.0, -0.5], rotation: [0, 0, 0], scale: 2, on: device)!,
//            TriangleGeometry(name: "plane-back", position: [0, 0, -1.5], rotation: [0, 0, 0], scale: 10, on: device)!
//        ]
    }
}
