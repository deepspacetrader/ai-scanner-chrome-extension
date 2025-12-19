console.log("Background service worker started")

// Listen for messages or install events
chrome.runtime.onInstalled.addListener(() => {
    console.log("Cyberpunk Scanner Installed")
})

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type === "CAPTURE_VISIBLE_TAB") {
        // null defaults to current window in the context of the action
        chrome.tabs.captureVisibleTab(
            null as any,
            { format: 'png' }
        ).then(dataUrl => {
            sendResponse({ dataUrl })
        }).catch(err => {
            console.error("Capture failed", err)
            sendResponse({ error: err.message })
        })
        return true // Keep channel open for async response
    }
})
