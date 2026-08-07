import Foundation
import CoreML
import Vision
import CoreVideo
import CoreImage
import CoreGraphics
import AppKit

struct Detection: Identifiable {
    let id = UUID()
    /// Assigned by ByteTracker; nil for raw detector output.
    var trackId: Int? = nil
    /// Normalized coords in the source image (0..1), origin top-left.
    let rect: CGRect
    let score: Float
    let classId: Int
    let className: String
    /// Optional grayscale mask (segmentation models). Sized to the detection box.
    let mask: CGImage?
}

/// Loads a YOLO26 CoreML model (Ultralytics export with NMS fused).
/// Input : image 640x640 RGB.
/// Detect output: MultiArray [1, 300, 6]  = (x1, y1, x2, y2, score, class) in input pixels (0..640).
/// Seg output:    MultiArray [1, 300, 38] = same 6 + 32 mask coefficients
///          plus  MultiArray [1, 32, 160, 160] = mask prototypes.
final class Detector {
    private let model: MLModel
    private let inputName: String
    private let outputName: String
    private let protoName: String?
    let inputSize: Int = 640
    let kind: ModelKind
    /// Class-id → human name, parsed from Ultralytics `names` metadata.
    let classNames: [Int: String]

    var confidenceThreshold: Float = 0.25

    func label(for classId: Int) -> String {
        classNames[classId] ?? "class \(classId)"
    }

    init(kind: ModelKind) throws {
        self.kind = kind
        let config = MLModelConfiguration()
        config.computeUnits = .all   // ANE + GPU + CPU

        guard let url = Bundle.main.url(forResource: kind.resourceName, withExtension: "mlmodelc") else {
            throw NSError(domain: "Detector", code: 1,
                          userInfo: [NSLocalizedDescriptionKey:
                            "Missing compiled model \(kind.resourceName).mlmodelc in bundle."])
        }
        self.model = try MLModel(contentsOf: url, configuration: config)
        self.inputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "image"

        var det: String?
        var proto: String?
        for (name, desc) in model.modelDescription.outputDescriptionsByName {
            guard let ma = desc.multiArrayConstraint else { continue }
            let shape = ma.shape.map { $0.intValue }
            if shape.count == 3, let last = shape.last, (last == 6 || last == 38) {
                det = name
            } else if shape.count == 4, shape.count >= 2, shape[1] == 32 {
                proto = name
            }
        }
        guard let outName = det else {
            let outs = model.modelDescription.outputDescriptionsByName.keys.joined(separator: ", ")
            throw NSError(domain: "Detector", code: 2,
                          userInfo: [NSLocalizedDescriptionKey:
                            "No detection output found for \(kind.resourceName). Outputs: \(outs)"])
        }
        self.outputName = outName
        self.protoName = proto

        // Parse Ultralytics `names` metadata, e.g. "{0: 'standing', 1: 'lying', 2: 'eating', ...}"
        let ud = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        self.classNames = Detector.parseNames(ud["names"])
    }

    private static func parseNames(_ raw: String?) -> [Int: String] {
        guard let raw else { return [:] }
        // Match  <digits>: '<name>'   or  <digits>: "<name>"
        let pattern = #"(\d+)\s*:\s*['"]([^'"]+)['"]"#
        guard let re = try? NSRegularExpression(pattern: pattern) else { return [:] }
        let ns = raw as NSString
        var out: [Int: String] = [:]
        re.enumerateMatches(in: raw, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
            guard let m, m.numberOfRanges == 3,
                  let id = Int(ns.substring(with: m.range(at: 1))) else { return }
            out[id] = ns.substring(with: m.range(at: 2))
        }
        return out
    }

    /// Runs detection on a CVPixelBuffer. Returns detections in normalized source coords.
    func detect(pixelBuffer: CVPixelBuffer) throws -> [Detection] {
        let srcW = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let srcH = CGFloat(CVPixelBufferGetHeight(pixelBuffer))

        let (letterboxed, scale, padX, padY) = letterbox(pixelBuffer: pixelBuffer, target: inputSize)

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(pixelBuffer: letterboxed)
        ])
        let out = try model.prediction(from: provider)
        guard let arr = out.featureValue(for: outputName)?.multiArrayValue else {
            return []
        }
        let proto: MLMultiArray? = protoName.flatMap { out.featureValue(for: $0)?.multiArrayValue }

        let n = arr.shape[1].intValue
        let rowLen = arr.shape[2].intValue     // 6 or 38
        let stride1 = arr.strides[1].intValue
        let stride2 = arr.strides[2].intValue
        let ptr = arr.dataPointer.bindMemory(to: Float32.self, capacity: arr.count)
        let hasMasks = (rowLen == 38 && proto != nil)

        var results: [Detection] = []
        results.reserveCapacity(n)
        for i in 0..<n {
            let base = i * stride1
            let score = ptr[base + 4 * stride2]
            if score < confidenceThreshold { continue }
            let x1 = CGFloat(ptr[base + 0 * stride2])
            let y1 = CGFloat(ptr[base + 1 * stride2])
            let x2 = CGFloat(ptr[base + 2 * stride2])
            let y2 = CGFloat(ptr[base + 3 * stride2])
            let cls = Int(ptr[base + 5 * stride2])

            // Un-letterbox to source-normalized.
            let sx1 = (x1 - padX) / scale
            let sy1 = (y1 - padY) / scale
            let sx2 = (x2 - padX) / scale
            let sy2 = (y2 - padY) / scale

            let nx = max(0, min(1, sx1 / srcW))
            let ny = max(0, min(1, sy1 / srcH))
            let nw = max(0, min(1 - nx, (sx2 - sx1) / srcW))
            let nh = max(0, min(1 - ny, (sy2 - sy1) / srcH))

            var mask: CGImage?
            if hasMasks, let proto {
                var coefs = [Float](repeating: 0, count: 32)
                for k in 0..<32 { coefs[k] = ptr[base + (6 + k) * stride2] }
                mask = buildMask(coefs: coefs, proto: proto,
                                 boxInInput: (x1, y1, x2, y2),
                                 tint: Detector.tintFor(classId: cls))
            }

            results.append(Detection(
                rect: CGRect(x: nx, y: ny, width: nw, height: nh),
                score: score, classId: cls,
                className: label(for: cls),
                mask: mask))
        }
        // Class-agnostic NMS: if two detections cover the same animal, keep only the
        // highest-confidence one. 0.30 lets closely-adjacent cows coexist.
        return Detector.nms(results, iouThreshold: 0.30)
    }

    static func nms(_ dets: [Detection], iouThreshold: Float) -> [Detection] {
        let sorted = dets.sorted { $0.score > $1.score }
        var keep: [Detection] = []
        for d in sorted {
            let dup = keep.contains { iou($0.rect, d.rect) > iouThreshold }
            if !dup { keep.append(d) }
        }
        return keep
    }

    private static func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let inter = a.intersection(b)
        if inter.isNull || inter.isEmpty { return 0 }
        let ia = inter.width * inter.height
        let ua = a.width * a.height + b.width * b.height - ia
        return ua > 0 ? Float(ia / ua) : 0
    }

    // MARK: - Segmentation mask

    static func tintFor(classId: Int) -> (UInt8, UInt8, UInt8) {
        let palette: [(UInt8, UInt8, UInt8)] = [
            (239,  68,  68),  // red
            (34, 197,  94),   // green
            (59, 130, 246),   // blue
            (234, 179,   8),  // yellow
            (168,  85, 247),  // purple
            (236,  72, 153),  // pink
            (20, 184, 166),   // teal
        ]
        return palette[abs(classId) % palette.count]
    }

    /// Build a pre-tinted RGBA CGImage for one detection (transparent outside the silhouette).
    /// Formula: mask(y,x) = sigmoid( Σ_k coef[k] * proto[k, y, x] ), thresholded at 0.5.
    /// Prototypes live in the letterboxed 640×640 input space at 160×160 (stride 4).
    private func buildMask(coefs: [Float],
                           proto: MLMultiArray,
                           boxInInput: (CGFloat, CGFloat, CGFloat, CGFloat),
                           tint: (UInt8, UInt8, UInt8)) -> CGImage? {
        let ph = proto.shape[2].intValue
        let pw = proto.shape[3].intValue
        let sK = proto.strides[1].intValue
        let sY = proto.strides[2].intValue
        let sX = proto.strides[3].intValue
        let pPtr = proto.dataPointer.bindMemory(to: Float32.self, capacity: proto.count)

        let strideRatio = CGFloat(inputSize) / CGFloat(pw)
        let mx1 = max(0, Int((boxInInput.0 / strideRatio).rounded(.down)))
        let my1 = max(0, Int((boxInInput.1 / strideRatio).rounded(.down)))
        let mx2 = min(pw, Int((boxInInput.2 / strideRatio).rounded(.up)))
        let my2 = min(ph, Int((boxInInput.3 / strideRatio).rounded(.up)))
        let w = mx2 - mx1
        let h = my2 - my1
        if w <= 0 || h <= 0 { return nil }

        // Step 1: binary mask (Ultralytics-style: sigmoid > 0.5).
        var mask = [Bool](repeating: false, count: w * h)
        for y in 0..<h {
            let my = my1 + y
            let rowBase = my * sY
            for x in 0..<w {
                let mx = mx1 + x
                let cellBase = rowBase + mx * sX
                var sum: Float = 0
                for k in 0..<32 {
                    sum += coefs[k] * pPtr[k * sK + cellBase]
                }
                let v = 1.0 / (1.0 + expf(-sum))
                mask[y * w + x] = v > 0.5
            }
        }

        // Step 2: contour = mask pixels that touch a non-mask pixel (Ultralytics `plot()` outline).
        var edge = [Bool](repeating: false, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                if !mask[y * w + x] { continue }
                let up    = y > 0     && mask[(y - 1) * w + x]
                let down  = y < h - 1 && mask[(y + 1) * w + x]
                let left  = x > 0     && mask[y * w + x - 1]
                let right = x < w - 1 && mask[y * w + x + 1]
                if !(up && down && left && right) { edge[y * w + x] = true }
            }
        }

        // Step 3: build premultiplied RGBA — light fill inside, solid outline on the edge.
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let fillA: Float = 90    // ~35% opacity fill
        let edgeA: Float = 255   // opaque outline
        for i in 0..<w * h {
            let a: Float
            if edge[i]      { a = edgeA }
            else if mask[i] { a = fillA }
            else            { continue }
            let idx = i * 4
            rgba[idx]     = UInt8(Float(tint.0) * a / 255)
            rgba[idx + 1] = UInt8(Float(tint.1) * a / 255)
            rgba[idx + 2] = UInt8(Float(tint.2) * a / 255)
            rgba[idx + 3] = UInt8(a)
        }

        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        let bmp: UInt32 = CGImageAlphaInfo.premultipliedLast.rawValue
                        | CGBitmapInfo.byteOrder32Big.rawValue
        return CGImage(
            width: w, height: h,
            bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: w * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: bmp),
            provider: provider,
            decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }

    // MARK: - Letterbox

    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    private func letterbox(pixelBuffer: CVPixelBuffer, target: Int) -> (CVPixelBuffer, CGFloat, CGFloat, CGFloat) {
        let srcW = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let srcH = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let t = CGFloat(target)

        let scale = min(t / srcW, t / srcH)
        let newW = srcW * scale
        let newH = srcH * scale
        let padX = (t - newW) / 2
        let padY = (t - newH) / 2

        var outBuf: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
        ]
        CVPixelBufferCreate(kCFAllocatorDefault, target, target,
                            kCVPixelFormatType_32BGRA,
                            attrs as CFDictionary, &outBuf)
        guard let dst = outBuf else { fatalError("Can't allocate letterbox buffer") }

        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        let scaled = ci.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
                       .transformed(by: CGAffineTransform(translationX: padX, y: padY))
        let bg = CIImage(color: CIColor(red: 114/255, green: 114/255, blue: 114/255))
                    .cropped(to: CGRect(x: 0, y: 0, width: t, height: t))
        let composed = scaled.composited(over: bg)

        ciContext.render(composed, to: dst,
                         bounds: CGRect(x: 0, y: 0, width: t, height: t),
                         colorSpace: CGColorSpaceCreateDeviceRGB())
        return (dst, scale, padX, padY)
    }
}
