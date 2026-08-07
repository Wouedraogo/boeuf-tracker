import Foundation
import CoreML

enum ModelKind: String, CaseIterable, Identifiable {
    case cbvd5Lr001   = "boeuf_cbvd5_lr001"
    case cbvd5Seg     = "boeuf_cbvd5_seg"
    case yolo26s640   = "boeuf_yolo26s_640"
    case yolo26s832   = "boeuf_yolo26s_832"
    /// COCO pre-filter model (not shown in the picker; used internally by CowFilter).
    case cocoYolo26n  = "yolo26n"

    static var pickable: [ModelKind] { [.cbvd5Lr001, .cbvd5Seg, .yolo26s640, .yolo26s832] }

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .cbvd5Lr001: return "CBVD5 · lr001 (detect)"
        case .cbvd5Seg:   return "CBVD5 · seg"
        case .yolo26s640: return "YOLO26s · 640"
        case .yolo26s832: return "YOLO26s · 832"
        case .cocoYolo26n: return "YOLO26n · COCO"
        }
    }

    /// Resource name inside the app bundle (without extension).
    /// After adding the .mlpackage to Xcode, it gets compiled to .mlmodelc.
    var resourceName: String { rawValue }

    var isSegmentation: Bool { self == .cbvd5Seg }
}
