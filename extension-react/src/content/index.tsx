import React from 'react'
import { createRoot, type Root } from 'react-dom/client'
import ScannerHUD from '../components/ScannerHUD'
import './content.css'

// Content script - minimal footprint until activated
const ROOT_ID = 'ai-scanner-tool-root'

let root: Root | null = null
let rootContainer: HTMLDivElement | null = null
let isInitialized = false

/**
 * Initialize the React root only when first activated
 */
function initializeRoot() {
    if (isInitialized) return

    // Create container with minimal impact
    rootContainer = document.createElement('div')
    rootContainer.id = ROOT_ID

    // Append to body
    document.body.appendChild(rootContainer)

    // Mount React
    root = createRoot(rootContainer)
    root.render(
        <React.StrictMode>
            <ScannerHUD />
        </React.StrictMode>
    )

    isInitialized = true
}

/**
 * Clean up the React root (for extension unload)
 */
function cleanup() {
    if (root) {
        root.unmount()
        root = null
    }
    if (rootContainer && rootContainer.parentNode) {
        rootContainer.parentNode.removeChild(rootContainer)
        rootContainer = null
    }
    isInitialized = false
}

// Initialize immediately but with zero visual impact
// The ScannerHUD component itself handles activation state
// and returns null when not active
initializeRoot()

// Cleanup on extension unload (if needed)
if (typeof chrome !== 'undefined' && chrome.runtime) {
    chrome.runtime.onSuspend?.addListener(cleanup)
}
