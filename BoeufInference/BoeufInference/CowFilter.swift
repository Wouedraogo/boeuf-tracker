import Foundation
import CoreML
import CoreVideo
import CoreGraphics

/// Cascade pre-filter: runs a generic COCO-trained YOLO26n first to isolate
/// real cows (COCO class 19), so the behaviour model (CBVD5) never scores
/// sheep, elephants, dogs, etc. — it just doesn't see them.
///
/// Bundle `yolo26n.mlpackage` in Xcode for this to activate. If the model is
/// missing the app still runs — the cascade quietly disables itself.
final class CowFilter {
    private let detector: Detector?
    /// COCO class ids we accept. 19 = cow. Add 20 (sheep) etc. to change scope.
    private let allowedCocoClasses: Set<Int> = [19]
    /// Minimum IoU between a behaviour detection and a cow detection to keep it.
    let minOverlap: Float = 0.35
    /// Minimum score from the COCO detector to trust a "cow" call.
    var cowConfidence: Float = 0.35

    /// nil detector means the cascade is a no-op.
    var isActive: Bool { detector != nil }

    init() {
        // Try to load a bundled COCO YOLO26n. If missing, the cascade is a no-op
        // and inference proceeds unchanged — exactly like before this class existed.
        if let _ = Bundle.main.url(forResource: "yolo26n", withExtension: "mlmodelc") {
            let kind = ModelKind.cocoYolo26n
            self.detector = try? Detector(kind: kind)
        } else {
            self.detector = nil
        }
        detector?.confidenceThreshold = cowConfidence
    }

    /// Filter `behaviourDetections` to keep only those whose box overlaps a cow
    /// box from the generic YOLO26n. If the filter is inactive, return input as-is.
    func filter(behaviour: [Detection], pixelBuffer: CVPixelBuffer) -> [Detection] {
        guard let detector else { return behaviour }
        let cows: [Detection]
        do {
            cows = try detector.detect(pixelBuffer: pixelBuffer)
                .filter { allowedCocoClasses.contains($0.classId) }
        } catch {
            return behaviour   // don't kill the pipeline on a filter failure
        }
        guard !cows.isEmpty else { return [] }   // no cows in frame → drop everything
        return behaviour.filter { b in
            cows.contains { iou($0.rect, b.rect) >= minOverlap }
        }
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let inter = a.intersection(b)
        if inter.isNull || inter.isEmpty { return 0 }
        let ia = inter.width * inter.height
        let ua = a.width * a.height + b.width * b.height - ia
        return ua > 0 ? Float(ia / ua) : 0
    }
}
