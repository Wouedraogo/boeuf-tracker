import SwiftUI
import AVFoundation
import AVKit
import CoreVideo

/// AVPlayer wrapper that also exposes the current frame as CVPixelBuffer
/// via an AVPlayerItemVideoOutput, so we can run inference on it.
@MainActor
final class VideoInferenceController: ObservableObject {
    @Published var detections: [Detection] = []
    @Published var fps: Double = 0
    @Published var isRunning: Bool = false
    @Published var videoSize: CGSize = .zero
    @Published var errorMessage: String?

    let player = AVPlayer()
    private var output: AVPlayerItemVideoOutput?
    private var displayLink: CVDisplayLink?
    private var detector: Detector?
    private let tracker = ByteTracker()
    private let cowFilter = CowFilter()
    private var inFlight = false

    func setConfidence(_ v: Float) {
        detector?.confidenceThreshold = v
    }
    private var lastInferenceAt = CFAbsoluteTimeGetCurrent()
    private var frameCount = 0
    private var lastFPSUpdate = CFAbsoluteTimeGetCurrent()
    private var statusObs: NSKeyValueObservation?
    private var loopObs: NSObjectProtocol?

    func load(url: URL, kind: ModelKind) {
        stop()
        errorMessage = nil
        do {
            let d = try Detector(kind: kind)
            detector = d
        } catch {
            errorMessage = error.localizedDescription
            return
        }
        tracker.reset()

        let asset = AVURLAsset(url: url)
        let item = AVPlayerItem(asset: asset)

        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        let out = AVPlayerItemVideoOutput(pixelBufferAttributes: attrs)
        item.add(out)
        self.output = out

        // Observe status so we surface AVPlayer failures (codec, sandbox, missing file...).
        statusObs = item.observe(\.status, options: [.new]) { [weak self] item, _ in
            Task { @MainActor in
                guard let self else { return }
                switch item.status {
                case .failed:
                    let base = item.error?.localizedDescription ?? "unknown"
                    let underlying = (item.error as NSError?)?
                        .userInfo[NSUnderlyingErrorKey] as? NSError
                    let extra = underlying.map { " (\($0.domain) \($0.code): \($0.localizedDescription))" } ?? ""
                    self.errorMessage = "AVPlayer: \(base)\(extra)"
                    NSLog("BoeufInference AVPlayer failed: \(base)\(extra)")
                case .readyToPlay:
                    self.errorMessage = nil
                default: break
                }
            }
        }
        // Loop on end.
        if let loopObs { NotificationCenter.default.removeObserver(loopObs) }
        loopObs = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime,
            object: item, queue: .main
        ) { [weak self] _ in
            guard let self else { return }
            self.player.seek(to: .zero)
            self.player.play()
        }

        player.replaceCurrentItem(with: item)

        Task { @MainActor in
            do {
                let tracks = try await asset.loadTracks(withMediaType: .video)
                if let track = tracks.first {
                    let size = try await track.load(.naturalSize)
                    let tx = try await track.load(.preferredTransform)
                    let applied = size.applying(tx)
                    self.videoSize = CGSize(width: abs(applied.width), height: abs(applied.height))
                }
            } catch { /* ignore */ }
        }

        startDisplayLink()
        player.play()
        isRunning = true
    }

    func stop() {
        player.pause()
        stopDisplayLink()
        detections = []
        isRunning = false
    }

    // MARK: - Display link

    private func startDisplayLink() {
        var link: CVDisplayLink?
        CVDisplayLinkCreateWithActiveCGDisplays(&link)
        guard let link else { return }
        let opaque = Unmanaged.passUnretained(self).toOpaque()
        CVDisplayLinkSetOutputCallback(link, { _, _, _, _, _, ctx in
            guard let ctx else { return kCVReturnSuccess }
            let ctrl = Unmanaged<VideoInferenceController>.fromOpaque(ctx).takeUnretainedValue()
            ctrl.tick()
            return kCVReturnSuccess
        }, opaque)
        CVDisplayLinkStart(link)
        self.displayLink = link
    }

    private func stopDisplayLink() {
        if let link = displayLink { CVDisplayLinkStop(link) }
        displayLink = nil
    }

    /// Called from the display-link thread. Grab a frame if available and dispatch inference.
    nonisolated private func tick() {
        Task { @MainActor in
            guard let output = self.output, let detector = self.detector else { return }
            let time = self.player.currentTime()
            guard output.hasNewPixelBuffer(forItemTime: time) else { return }
            guard let pb = output.copyPixelBuffer(forItemTime: time, itemTimeForDisplay: nil) else { return }
            if self.inFlight { return }
            self.inFlight = true

            let start = CFAbsoluteTimeGetCurrent()
            let filter = self.cowFilter
            Task.detached(priority: .userInitiated) {
                var dets: [Detection] = []
                do {
                    let raw = try detector.detect(pixelBuffer: pb)
                    // Cascade: keep only detections that overlap a real "cow"
                    // (COCO class 19). No-op if yolo26n.mlpackage isn't bundled.
                    dets = filter.filter(behaviour: raw, pixelBuffer: pb)
                } catch {
                    await MainActor.run { self.errorMessage = error.localizedDescription }
                }
                await MainActor.run {
                    // Pass filtered detections through ByteTracker → stable IDs +
                    // majority-voted class labels + gap filling on missed frames.
                    self.detections = self.tracker.update(dets)
                    self.inFlight = false
                    self.frameCount += 1
                    let now = CFAbsoluteTimeGetCurrent()
                    _ = start
                    if now - self.lastFPSUpdate > 0.5 {
                        self.fps = Double(self.frameCount) / (now - self.lastFPSUpdate)
                        self.frameCount = 0
                        self.lastFPSUpdate = now
                    }
                }
            }
        }
    }
}

/// SwiftUI wrapper for AVPlayerView.
struct PlayerLayerView: NSViewRepresentable {
    let player: AVPlayer
    func makeNSView(context: Context) -> AVPlayerView {
        let v = AVPlayerView()
        v.player = player
        v.controlsStyle = .floating
        v.showsFullScreenToggleButton = true
        return v
    }
    func updateNSView(_ nsView: AVPlayerView, context: Context) {
        nsView.player = player
    }
}

/// Overlay drawing detection boxes on top of the video, scaled to the current view rect.
struct DetectionOverlay: View {
    let detections: [Detection]
    let videoSize: CGSize

    var body: some View {
        GeometryReader { geo in
            let viewRect = geo.frame(in: .local)
            let fit = aspectFit(video: videoSize, in: viewRect.size)
            ZStack {
                ForEach(detections) { d in
                    let r = CGRect(
                        x: fit.origin.x + d.rect.origin.x * fit.width,
                        y: fit.origin.y + d.rect.origin.y * fit.height,
                        width: d.rect.width * fit.width,
                        height: d.rect.height * fit.height)
                    // Color is derived from track id so each animal keeps the same
                    // color across the whole video (like ByteTrack's demo output).
                    let c = color(for: d.trackId ?? d.classId)
                    let label = d.trackId.map { "#\($0) \(d.className)" } ?? d.className
                    if let mask = d.mask {
                        Image(decorative: mask, scale: 1)
                            .resizable()
                            .interpolation(.medium)
                            .frame(width: r.width, height: r.height)
                            .position(x: r.midX, y: r.midY)
                            .allowsHitTesting(false)
                        Text(label)
                            .font(.system(size: 11, weight: .semibold))
                            .padding(.horizontal, 4).padding(.vertical, 1)
                            .background(c.opacity(0.85))
                            .foregroundStyle(.white)
                            .position(x: r.midX, y: max(10, r.minY - 8))
                    } else {
                        Rectangle()
                            .path(in: r)
                            .stroke(c, lineWidth: 2)
                        Text(label)
                            .font(.system(size: 11, weight: .semibold))
                            .padding(.horizontal, 4).padding(.vertical, 1)
                            .background(c.opacity(0.85))
                            .foregroundStyle(.white)
                            .position(x: r.minX + 40, y: max(8, r.minY - 8))
                    }
                }
            }
        }
        .allowsHitTesting(false)
    }

    private func aspectFit(video: CGSize, in container: CGSize) -> CGRect {
        guard video.width > 0, video.height > 0 else { return .init(origin: .zero, size: container) }
        let s = min(container.width / video.width, container.height / video.height)
        let w = video.width * s
        let h = video.height * s
        return CGRect(x: (container.width - w) / 2, y: (container.height - h) / 2, width: w, height: h)
    }

    private func color(for cls: Int) -> Color {
        let palette: [Color] = [.red, .green, .blue, .orange, .purple, .pink, .yellow, .cyan]
        return palette[cls % palette.count]
    }
}
