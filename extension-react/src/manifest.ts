import { defineManifest } from '@crxjs/vite-plugin'

export default defineManifest({
    manifest_version: 3,
    name: "AI Scanner Tool",
    version: "1.0.0",
    description: "AI-enhanced image analysis and text summarization with a Cyberpunk styled interface.",
    permissions: ["activeTab", "storage", "downloads"],
    host_permissions: ["<all_urls>"],
    action: {
        default_popup: "index.html",
    },
    content_scripts: [
        {
            matches: ["<all_urls>"],
            js: ["src/content/index.tsx"],
            run_at: "document_end",
        },
    ],
    background: {
        service_worker: "src/background/index.ts",
        type: "module",
    },
    web_accessible_resources: [
        {
            resources: ["*.pt", "*.png", "*.jpg"],
            matches: ["<all_urls>"],
        },
    ],
})
