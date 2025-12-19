export interface DetectionBox {
    class: string
    confidence: number
    bbox: {
        x1: number
        y1: number
        x2: number
        y2: number
    }
}

export interface DetectionResult {
    type: 'detection'
    data: {
        x: number
        y: number
        width: number
        height: number
        color: string
        type: string
        analysis?: string
        confidence: number
    }[]
    total_objects: number
    image?: string
}

export class ImageScannerService {
    private detectionCache = new Map<string, DetectionResult>()
    private cacheMaxSize = 50
    private failedDetectionCooldowns = new Map<string, number>()
    private detectionRetryDelayMs = 1500
    private isProcessing = false

    // Settings - could be injectable
    private settings = {
        detectionEndpoint: "http://localhost:8001/api/detect-base64",
        showCrawlingLines: true,
    }

    constructor() { }

    public getCachedDetection(srcKey: string): DetectionResult | undefined {
        return this.detectionCache.get(srcKey)
    }

    public getImageSourceKey(img: HTMLImageElement | Element | null): string {
        if (!img) return ""

        // Check if it's an IMG element
        if (img instanceof HTMLImageElement) {
            return img.currentSrc || img.src || ""
        }

        // Fallback for background images
        // Note: We might need to handle this differently in React, but sticking to logic
        if ((img as any).__imageAIHoverSrc) {
            return (img as any).__imageAIHoverSrc
        }

        return ""
    }

    public async detectImage(img: HTMLImageElement | Element, save: boolean = false): Promise<DetectionResult | null> {
        const srcKey = this.getImageSourceKey(img)
        if (!srcKey) return null

        // Check cooling down
        if (this.shouldRespectFailureCooldown(srcKey)) return null

        // Check cache
        if (this.detectionCache.has(srcKey)) {
            return this.detectionCache.get(srcKey) || null
        }

        if (this.isProcessing) return null
        this.isProcessing = true

        try {
            const base64 = await this.imageToBase64(img, save)
            let result: DetectionResult | null = null

            if (base64) {
                result = await this.fetchDetection(base64, srcKey, save)
            } else {
                // Fallback to URL detection
                result = await this.fetchDetectionByUrl(srcKey)
            }

            if (result && result.data && result.data.length > 0) {
                result.image = base64 || undefined
                this.cacheResult(srcKey, result)
                return result
            } else {
                this.scheduleFailureCooldown(srcKey)
                return null
            }
        } catch (e) {
            console.error(e)
            this.scheduleFailureCooldown(srcKey)
            return null
        } finally {
            this.isProcessing = false
        }
    }

    private async fetchDetection(base64: string, _key: string, save: boolean = false): Promise<DetectionResult | null> {
        try {
            const response = await fetch(this.settings.detectionEndpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: base64, save })
            })
            if (!response.ok) throw new Error("API Error")
            return await response.json()
        } catch (e) {
            console.error("Detection API error", e)
            return null
        }
    }

    public async analyzeBox(originalBase64: string, x: number, y: number, width: number, height: number, type: string = "person"): Promise<string | null> {
        try {
            const endpoint = this.settings.detectionEndpoint.replace("/api/detect-base64", "/api/analyze-box")
            const response = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    image: originalBase64,
                    box: { x, y, width, height },
                    type
                })
            })

            if (!response.ok) return "Analysis failed"
            const data = await response.json()
            return data.analysis || "No result"
        } catch (e) {
            console.error("Analyze box error", e)
            return "Analysis error"
        }
    }

    private async fetchDetectionByUrl(url: string): Promise<DetectionResult | null> {
        const endpoint = this.settings.detectionEndpoint.replace("/api/detect-base64", "/api/detect-url")
        try {
            const response = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url })
            })
            if (!response.ok) throw new Error("API Error")
            return await response.json()
        } catch (e) {
            console.error("URL Detection API error", e)
            return null
        }
    }

    private cacheResult(key: string, result: DetectionResult) {
        this.detectionCache.set(key, result)
        if (this.detectionCache.size > this.cacheMaxSize) {
            const first = this.detectionCache.keys().next().value
            if (first) this.detectionCache.delete(first)
        }
    }

    private shouldRespectFailureCooldown(key: string): boolean {
        const cooldown = this.failedDetectionCooldowns.get(key)
        if (!cooldown) return false
        if (Date.now() >= cooldown) {
            this.failedDetectionCooldowns.delete(key)
            return false
        }
        return true
    }

    private scheduleFailureCooldown(key: string) {
        this.failedDetectionCooldowns.set(key, Date.now() + this.detectionRetryDelayMs)
    }

    /**
     * Convert image to base64 using the already-loaded DOM element.
     * This uses the browser's cache - NO re-download from server.
     */
    private imageToBase64(img: HTMLImageElement | Element, save: boolean = false): Promise<string | null> {
        return new Promise((resolve) => {
            try {
                // If it's an HTMLImageElement, draw it directly to canvas
                if (img instanceof HTMLImageElement) {
                    // Check if image is loaded
                    if (!img.complete || img.naturalWidth === 0) {
                        // Image not loaded yet, wait for it
                        img.onload = () => {
                            resolve(this.drawImageToBase64(img))
                        }
                        img.onerror = () => resolve(null)
                        return
                    }

                    // Image already loaded - use it directly from DOM (browser cache)
                    const base64 = this.drawImageToBase64(img)
                    if (base64) {
                        resolve(base64)
                        return
                    }

                    // If tainted canvas (CORS), try capturing visible tab and cropping
                    this.captureAndCrop(img, save).then(resolve)
                    return
                }

                // For background images or other elements, we may need to fetch
                const srcKey = this.getImageSourceKey(img)
                if (!srcKey) {
                    resolve(null)
                    return
                }

                // Data URLs - already have the data
                if (srcKey.startsWith("data:")) {
                    resolve(srcKey.split(",")[1])
                    return
                }

                // Fallback: create new image (will use browser HTTP cache if available)
                const newImg = new Image()
                newImg.crossOrigin = "anonymous"
                newImg.src = srcKey

                newImg.onload = () => {
                    resolve(this.drawImageToBase64(newImg))
                }
                newImg.onerror = () => resolve(null)
            } catch (e) {
                resolve(null)
            }
        })
    }

    /**
     * Draw an image element to canvas and get base64.
     * Uses the browser's cached image data.
     */
    private drawImageToBase64(img: HTMLImageElement): string | null {
        try {
            const canvas = document.createElement("canvas")
            canvas.width = img.naturalWidth || img.width
            canvas.height = img.naturalHeight || img.height
            const ctx = canvas.getContext("2d")
            if (!ctx) return null
            ctx.drawImage(img, 0, 0)
            const dataUrl = canvas.toDataURL("image/jpeg", 0.85)
            return dataUrl.split(",")[1]
        } catch (e) {
            // CORS or tainted canvas error
            return null
        }
    }

    /**
     * Fallback: Capture the current tab's viewport and crop the image from it.
     * This bypasses CORS restrictions/tainted canvas by using the visible pixels.
     */
    private async captureAndCrop(img: Element, save: boolean = false): Promise<string | null> {
        try {
            const rect = img.getBoundingClientRect()
            if (rect.width === 0 || rect.height === 0) return null

            // Check if chrome runtime is available
            if (typeof chrome === 'undefined' || !chrome.runtime) return null

            // Hide UI elements if we are actually saving the image to disk
            // This prevents the reticle and HUD from dirtying the resulting file
            const reticle = document.getElementById('scanner-reticle')
            const hud = document.getElementById('ai-scanner-hud-container')

            if (save) {
                if (reticle) reticle.style.display = 'none'
                if (hud) hud.style.display = 'none'
            }

            let response: any;
            try {
                // Send message to background to capture tab
                response = await chrome.runtime.sendMessage({ type: "CAPTURE_VISIBLE_TAB" })
            } finally {
                // Restore visibility
                if (save) {
                    if (reticle) reticle.style.display = 'block'
                    if (hud) hud.style.display = 'block'
                }
            }

            if (!response || !response.dataUrl) return null

            return new Promise((resolve) => {
                const image = new Image()
                image.onload = () => {
                    const canvas = document.createElement("canvas")
                    // Handle scaling (HiDPI screens)
                    // captureVisibleTab returns the full physical pixels image
                    const scale = image.width / window.innerWidth

                    // We can output at high res or logical size.
                    // Let's output at logical size (rect size) * dpr approx?
                    // Or just use the scale we found to get maximum quality.
                    canvas.width = rect.width * scale
                    canvas.height = rect.height * scale

                    const ctx = canvas.getContext("2d")
                    if (!ctx) {
                        resolve(null)
                        return
                    }

                    // Draw the crop
                    // args: image, sx, sy, sw, sh, dx, dy, dw, dh
                    ctx.drawImage(
                        image,
                        rect.x * scale, rect.y * scale, rect.width * scale, rect.height * scale,
                        0, 0, canvas.width, canvas.height
                    )

                    const dataUrl = canvas.toDataURL("image/jpeg", 0.85)
                    resolve(dataUrl.split(",")[1])
                }
                image.onerror = () => resolve(null)
                image.src = response.dataUrl
            })
        } catch (e) {
            console.error("Capture crop failed", e)
            return null
        }
    }
}

export const imageScanner = new ImageScannerService()
