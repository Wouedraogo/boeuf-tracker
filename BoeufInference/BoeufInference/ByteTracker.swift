import Foundation
import CoreGraphics

/// Multi-object tracker inspired by ByteTrack (Zhang et al., ECCV 2022).
/// https://github.com/ifzhang/ByteTrack
///
/// Simplifications vs the paper:
/// - Motion model = constant-velocity on the box center (poor-man's Kalman), no
///   full covariance. Good enough for slow-moving cattle at 20+ FPS.
/// - Data association = greedy IoU (rather than Hungarian). Optimal for < ~50
///   objects per frame, which is our case.
/// - Two-stage association (high-conf then low-conf) preserved.
final class ByteTracker {
    private var tracks: [Track] = []
    private var nextId: Int = 1

    /// Params (tunable).
    let highScoreThresh: Float = 0.5
    let lowScoreThresh:  Float = 0.10
    /// Min IoU to accept a match in either stage.
    let matchIoU:  Float = 0.30
    /// Frames without a matching detection before a track is dropped.
    let maxAge:    Int = 30
    /// Consecutive hits required before a track is reported to the UI.
    let minHits:   Int = 3
    /// Rolling window (in frames) over which we vote the reported class.
    /// Small = reactive (walking/eating changes surface fast). Large = stable.
    let classVoteWindow: Int = 12

    func reset() {
        tracks.removeAll()
        nextId = 1
    }

    /// Called once per frame with the raw detector output. Returns the smoothed,
    /// class-voted, ID-carrying detections that should be drawn.
    func update(_ dets: [Detection]) -> [Detection] {
        // 1. Predict every track one step forward.
        for t in tracks { t.predict() }

        // 2. Split incoming detections by confidence.
        let high = dets.filter { $0.score >= highScoreThresh }
        var low  = dets.filter { $0.score >= lowScoreThresh && $0.score < highScoreThresh }

        // 3. First association: high-conf ↔ all tracks.
        var unmatchedTracks = tracks
        var unmatchedHigh = high
        greedyMatch(dets: &unmatchedHigh, tracks: &unmatchedTracks, minIoU: matchIoU)

        // 4. Second association: low-conf ↔ tracks still unmatched.
        greedyMatch(dets: &low, tracks: &unmatchedTracks, minIoU: matchIoU)

        // 5. Age unmatched tracks.
        for t in unmatchedTracks { t.timeSinceUpdate += 1 }

        // 6. Spawn new tentative tracks from the leftover high-confidence detections.
        for d in unmatchedHigh {
            tracks.append(Track(id: nextId, detection: d, classWindow: classVoteWindow))
            nextId += 1
        }

        // 7. Retire tracks that have been unseen too long.
        tracks.removeAll { $0.timeSinceUpdate > maxAge }

        // 8. Emit confirmed tracks with smoothed geometry and majority class.
        return tracks.compactMap { t in
            guard t.hits >= minHits else { return nil }
            let cls = t.majorityClass()
            let name = t.className(for: cls)
            return Detection(
                trackId: t.id,
                rect: t.currentRect(),
                score: t.lastScore,
                classId: cls,
                className: name,
                mask: t.lastMask)
        }
    }

    /// Greedy IoU association: sort (track, det) pairs by IoU desc, take highest
    /// unclaimed pair, repeat. Updates matched tracks in place and removes
    /// matched entries from the caller's arrays.
    private func greedyMatch(dets: inout [Detection], tracks: inout [Track], minIoU: Float) {
        struct Pair { let ti: Int; let di: Int; let iou: Float }
        var pairs: [Pair] = []
        pairs.reserveCapacity(dets.count * tracks.count)
        for (ti, t) in tracks.enumerated() {
            let tr = t.currentRect()
            for (di, d) in dets.enumerated() {
                let v = iou(tr, d.rect)
                if v >= minIoU { pairs.append(Pair(ti: ti, di: di, iou: v)) }
            }
        }
        pairs.sort { $0.iou > $1.iou }

        var usedT = Set<Int>()
        var usedD = Set<Int>()
        for p in pairs {
            if usedT.contains(p.ti) || usedD.contains(p.di) { continue }
            tracks[p.ti].update(with: dets[p.di])
            usedT.insert(p.ti)
            usedD.insert(p.di)
        }

        dets   = dets.enumerated().filter   { !usedD.contains($0.offset) }.map   { $0.element }
        tracks = tracks.enumerated().filter { !usedT.contains($0.offset) }.map   { $0.element }
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let inter = a.intersection(b)
        if inter.isNull || inter.isEmpty { return 0 }
        let ia = inter.width * inter.height
        let ua = a.width * a.height + b.width * b.height - ia
        return ua > 0 ? Float(ia / ua) : 0
    }
}

// MARK: - Track

final class Track {
    let id: Int

    // Motion state (normalized coords).
    private var cx: CGFloat, cy: CGFloat
    private var w: CGFloat,  h: CGFloat
    private var vcx: CGFloat = 0
    private var vcy: CGFloat = 0

    // Bookkeeping.
    var hits: Int = 1
    var timeSinceUpdate: Int = 0
    var lastScore: Float
    var lastMask: CGImage?

    // Class voting.
    private var classHistory: [(cls: Int, score: Float)] = []
    private var classNames: [Int: String] = [:]
    private let classWindow: Int

    init(id: Int, detection d: Detection, classWindow: Int) {
        self.id = id
        self.cx = d.rect.midX
        self.cy = d.rect.midY
        self.w  = d.rect.width
        self.h  = d.rect.height
        self.lastScore = d.score
        self.lastMask  = d.mask
        self.classWindow = classWindow
        classHistory.append((d.classId, d.score))
        classNames[d.classId] = d.className
    }

    /// Move the state one frame forward with the current velocity.
    func predict() {
        cx += vcx
        cy += vcy
        // Clamp inside the frame so a lost track doesn't drift off screen.
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
    }

    /// Fold in a matching detection: refresh geometry, mask, class vote.
    func update(with d: Detection) {
        let newCx = d.rect.midX
        let newCy = d.rect.midY
        let alpha: CGFloat = 0.6   // EMA on velocity
        vcx = alpha * (newCx - cx) + (1 - alpha) * vcx
        vcy = alpha * (newCy - cy) + (1 - alpha) * vcy
        cx = newCx
        cy = newCy
        // Smooth size a bit so it doesn't jitter frame-to-frame.
        let sz: CGFloat = 0.5
        w = sz * d.rect.width  + (1 - sz) * w
        h = sz * d.rect.height + (1 - sz) * h

        hits += 1
        timeSinceUpdate = 0
        lastScore = d.score
        lastMask  = d.mask
        classHistory.append((d.classId, d.score))
        classNames[d.classId] = d.className
        if classHistory.count > classWindow {
            classHistory.removeFirst(classHistory.count - classWindow)
        }
    }

    func currentRect() -> CGRect {
        CGRect(x: cx - w / 2, y: cy - h / 2, width: w, height: h)
    }

    /// Class with the highest **recency-weighted** cumulative confidence.
    /// Recent frames dominate: weight = 1.5^i where i counts from oldest → newest.
    /// This lets the label flip to `walking` after ~4-5 consecutive walking frames
    /// even if the window is filled with older `standing` votes.
    func majorityClass() -> Int {
        var acc: [Int: Float] = [:]
        let n = classHistory.count
        for (i, entry) in classHistory.enumerated() {
            let weight: Float = powf(1.5, Float(i - n + 1))   // newest = 1, older shrinks
            acc[entry.cls, default: 0] += entry.score * weight
        }
        return acc.max { $0.value < $1.value }?.key ?? classHistory.last!.cls
    }

    func className(for cls: Int) -> String {
        classNames[cls] ?? "class \(cls)"
    }
}
