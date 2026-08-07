import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var kind: ModelKind = .yolo26s640
    @State private var folderURL: URL?
    @State private var videos: [URL] = []
    @State private var selectedVideo: URL?
    @State private var confidence: Double = 0.25
    @StateObject private var ctrl = VideoInferenceController()
    /// Kept alive so security-scoped access to the picked folder stays open.
    @State private var scopedFolder: URL?

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 260, ideal: 300)
        } detail: {
            detail
        }
        .onChange(of: selectedVideo) { _, new in
            guard let new else { return }
            ctrl.load(url: new, kind: kind)
            ctrl.setConfidence(Float(confidence))
        }
        .onChange(of: kind) { _, _ in
            if let v = selectedVideo {
                ctrl.load(url: v, kind: kind)
                ctrl.setConfidence(Float(confidence))
            }
        }
        .onChange(of: confidence) { _, new in
            ctrl.setConfidence(Float(new))
        }
        .onAppear {
            ctrl.setConfidence(Float(confidence))
            // Reopen the last folder used in the previous session.
            if folderURL == nil,
               let saved = UserDefaults.standard.string(forKey: "lastFolderPath") {
                let url = URL(fileURLWithPath: saved, isDirectory: true)
                if FileManager.default.fileExists(atPath: url.path) {
                    folderURL = url
                    loadVideos(in: url)
                }
            }
        }
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Modèle").font(.headline)
            Picker("", selection: $kind) {
                ForEach(ModelKind.pickable) { k in
                    Text(k.displayName).tag(k)
                }
            }
            .pickerStyle(.inline)
            .labelsHidden()

            Divider()

            HStack {
                Text("Confiance").font(.headline)
                Spacer()
                Text(String(format: "%.2f", confidence)).monospacedDigit()
            }
            Slider(value: $confidence, in: 0.05...0.9)

            Divider()

            HStack {
                Text("Dossier vidéos").font(.headline)
                Spacer()
                Button("Choisir…") { pickFolder() }
            }
            if let folderURL {
                Text(folderURL.path).font(.caption).foregroundStyle(.secondary)
                    .lineLimit(2).truncationMode(.middle)
                Text("\(videos.count) vidéo(s) trouvée(s)")
                    .font(.caption2).foregroundStyle(videos.isEmpty ? .red : .secondary)
            }

            List(videos, id: \.self, selection: $selectedVideo) { url in
                Text(url.lastPathComponent).lineLimit(1).truncationMode(.middle)
                    .tag(Optional(url))
            }
            .frame(maxHeight: .infinity)
        }
        .padding(12)
    }

    // MARK: - Detail

    private var detail: some View {
        VStack(spacing: 0) {
            ZStack {
                Color.black
                PlayerLayerView(player: ctrl.player)
                DetectionOverlay(detections: ctrl.detections, videoSize: ctrl.videoSize)
            }
            statusBar
        }
    }

    private var statusBar: some View {
        HStack(spacing: 16) {
            Label(kind.displayName, systemImage: "cpu")
            Label(String(format: "%.1f inf/s", ctrl.fps), systemImage: "speedometer")
            Label("\(ctrl.detections.count) détections", systemImage: "square.dashed")
            Spacer()
            if let msg = ctrl.errorMessage {
                Label(msg, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.red)
            }
        }
        .font(.caption)
        .padding(8)
        .background(.thinMaterial)
    }

    // MARK: - Folder picking

    private func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Choisir"
        if panel.runModal() == .OK, let url = panel.url {
            // Release previous scope, then start a new one and keep it open.
            if let old = scopedFolder { old.stopAccessingSecurityScopedResource() }
            let ok = url.startAccessingSecurityScopedResource()
            scopedFolder = ok ? url : nil
            folderURL = url
            loadVideos(in: url)
            // Remember the folder for next launch (sandbox is off → path alone is enough).
            UserDefaults.standard.set(url.path, forKey: "lastFolderPath")
        }
    }

    private func loadVideos(in url: URL) {
        let exts: Set<String> = ["mp4", "mov", "m4v", "avi", "mkv"]
        let fm = FileManager.default
        // If a file was picked (Xcode 27 panel quirk), fall back to its parent.
        var dir = url
        var isDir: ObjCBool = false
        if fm.fileExists(atPath: url.path, isDirectory: &isDir), !isDir.boolValue {
            dir = url.deletingLastPathComponent()
            folderURL = dir
        }
        let items = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        videos = items
            .filter { exts.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        // If original pick was a specific file, preselect it.
        if !isDir.boolValue, videos.contains(url) {
            selectedVideo = url
        } else {
            selectedVideo = videos.first
        }
    }
}
