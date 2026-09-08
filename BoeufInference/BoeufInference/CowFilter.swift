import Foundation
import CoreML
import CoreVideo
import CoreGraphics

/// Two-stage cascade wrapping the generic COCO YOLO26n:
///
/// - **Detection**  : COCO is much better at finding cows in cluttered / occluded
///   scenes than a CBVD5 model trained only on cattle-behaviour footage.
/// - **Behaviour**  : CBVD5 knows the 7 activity classes but sometimes misses a
///   partially-hidden animal.
///
/// So we run BOTH per frame and merge:
///   1. Start from the COCO cow boxes (better geometry, better recall).
///   2. For each COCO box, find the CBVD5 detection with highest IoU and
///      inherit its class + segmentation mask (if any).
///   3. If no CBVD5 detection matches, keep the box with class = `other`.
///
/// Bundle `yolo26n.mlpackage` in Xcode. If it's missing the cascade quietly
/// disables itself and the app falls back to CBVD5-only detection.
final class CowFilter {
    private let detector: Detector?
    private let allowedCocoClasses: Set<Int> = [19]           // 19 = cow
    private let matchIoU: Float = 0.30
    /// COCO detection confidence threshold. Kept low so occluded cows survive.
    var cowConfidence: Float = 0.30
    /// Fallback class id when a COCO cow has no matching behaviour detection.
    /// 6 = "other" in the CBVD5 label map. Displayed as "?" so the user knows
    /// the behaviour is temporarily unknown (the tracker's class vote will
    /// converge to the real label after a few frames).
    private let fallbackClassId = 6
    private let fallbackClassName = "?"

    var isActive: Bool { detector != nil }

    init() {
        if Bundle.main.url(forResource: "yolo26n", withExtension: "mlmodelc") != nil {
            self.detector = try? Detector(kind: .cocoYolo26n)
        } else {
            self.detector = nil
        }
        detector?.confidenceThreshold = cowConfidence
    }

    /// Merge COCO-detected cow boxes with CBVD5 behaviour labels.
    /// Returns COCO's box geometry + CBVD5's class/mask when they align.
    ///
    /// If the COCO model isn't bundled, returns `behaviour` unchanged so the
    /// legacy CBVD5-only pipeline keeps working.
    func merge(behaviour: [Detection], pixelBuffer: CVPixelBuffer) -> [Detection] {
        guard let detector else { return behaviour }
        let cows: [Detection]
        do {
            cows = try detector.detect(pixelBuffer: pixelBuffer)
                .filter { allowedCocoClasses.contains($0.classId) }
        } catch {
            return behaviour                        // don't kill the pipeline on failure
        }
        if cows.isEmpty { return [] }               // no cows visible → drop everything

        return cows.map { cow -> Detection in
            // Best-IoU behaviour detection for this cow, if any is close enough.
            var bestI: Float = 0
            var best: Detection?
            for b in behaviour {
                let v = iou(cow.rect, b.rect)
                if v > bestI { bestI = v; best = b }
            }
            if let b = best, bestI >= matchIoU {
                // Inherit CBVD5 class + mask, but keep COCO's tighter box.
                // Mask is aligned to `b.rect`, not `cow.rect` — keep the mask
                // only if the two rects are close enough that the mask still
                // lines up visually.
                let keepMask = bestI >= 0.6
                return Detection(
                    trackId: nil,
                    rect: cow.rect,
                    score: max(cow.score, b.score),
                    classId: b.classId,
                    className: b.className,
                    mask: keepMask ? b.mask : nil)
            }
            // COCO saw a cow the behaviour model missed → keep the box, no class info.
            return Detection(
                trackId: nil,
                rect: cow.rect,
                score: cow.score,
                classId: fallbackClassId,
                className: fallbackClassName,
                mask: nil)
        }
    }

    // Backwards-compatible name kept in case anything else calls it.
    func filter(behaviour: [Detection], pixelBuffer: CVPixelBuffer) -> [Detection] {
        merge(behaviour: behaviour, pixelBuffer: pixelBuffer)
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let inter = a.intersection(b)
        if inter.isNull || inter.isEmpty { return 0 }
        let ia = inter.width * inter.height
        let ua = a.width * a.height + b.width * b.height - ia
        return ua > 0 ? Float(ia / ua) : 0
    }
}
