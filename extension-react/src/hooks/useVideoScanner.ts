import { useState, useEffect, useCallback, useRef } from 'react'
import { imageScanner, type DetectionResult } from '../services/imageScanner'
import { type SummarySettings, textSummarizer } from '../services/textSummarizer'

export function useVideoScanner(
    isActive: boolean,
    isHoveringSidebar: boolean = false,
    saveScannedImages: boolean = false,
    enableDeepAnalysis: boolean = false,
    enableEnhancedDescription: boolean = true,
    deepAnalysisThreshold: number = 0.85,
    categoryThresholds: Record<string, number> = {},
    summarySettings?: SummarySettings,
    visionModel: string = 'florence2',
    lmStudioUrl: string = 'http://localhost:1234'
) {
    const [hoveredVideo, setHoveredVideo] = useState<HTMLVideoElement | null>(null)
    const hoveredVideoRef = useRef<HTMLVideoElement | null>(null)
    const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null)
    const [isScanning, setIsScanning] = useState(false)
    const [mousePos, setMousePos] = useState({ x: 0, y: 0 })
    const analyzingRef = useRef<Set<string>>(new Set())
    const clearTimeoutRef = useRef<NodeJS.Timeout | null>(null)
    // Cache for detection results with analysis
    const detectionCacheRef = useRef<Map<string, DetectionResult>>(new Map())

    // Find video under cursor
    const findVideoUnderCursor = useCallback((e: MouseEvent) => {
        const elements = document.elementsFromPoint(e.clientX, e.clientY)
        for (const el of elements) {
            if (el instanceof HTMLVideoElement) {
                // Additional check for YouTube: ensure we're not hovering over the controls
                const videoContainer = el.closest('.html5-video-container')
                if (videoContainer) {
                    const controls = videoContainer.querySelector('.ytp-chrome-bottom')
                    if (controls?.contains(e.target as Node)) {
                        return null
                    }
                }
                return el
            }
        }
        return null
    }, [])



    const triggerDeepAnalysis = useCallback(async (result: DetectionResult) => {
        if (!result.image) return

        const targets = result.data.filter(d => {
            const category = (d as any).category || "Misc"
            const threshold = categoryThresholds[category] ?? deepAnalysisThreshold
            const isAnalyzable = (d as any).is_analyzable
            const analysisPending = (d as any).analysis_pending
            // Include objects with analysis_pending flag or empty analysis
            return isAnalyzable && d.confidence >= threshold && (analysisPending || !d.analysis)
        })

        if (targets.length === 0) return

        const analyzeKey = `${result.image.substring(0, 50)}_${targets.length}`
        if (analyzingRef.current.has(analyzeKey)) return
        analyzingRef.current.add(analyzeKey)

        setDetectionResult(prev => {
            if (!prev) return prev
            return {
                ...prev,
                data: prev.data.map(d => {
                    const category = (d as any).category || "Misc"
                    const threshold = categoryThresholds[category] ?? deepAnalysisThreshold
                    const isAnalyzable = (d as any).is_analyzable
                    return (isAnalyzable && d.confidence >= threshold && ((d as any).analysis_pending || !d.analysis)) ? { ...d, analysis: '...' } : d
                })
            }
        })

        try {
            await Promise.all(targets.map(async (target) => {
                const visionResult = await imageScanner.analyzeBox(
                    result.image!,
                    target.x,
                    target.y,
                    target.width,
                    target.height,
                    target.type,
                    visionModel,
                    visionModel,
                    lmStudioUrl
                )

                if (!visionResult) return

                let finalAnalysis = visionResult.analysis
                let analysisModel = visionResult.model || visionModel

                const isAnalyzable = (target as any).is_analyzable
                const isHighFidelity = analysisModel.includes('Samsung') || analysisModel.includes('Qwen2-VL') || visionModel.includes('gemma') || visionModel.includes('LM Studio')

                if (enableEnhancedDescription && isAnalyzable && visionResult && !visionResult.analysis.startsWith('Error') && summarySettings && !isHighFidelity) {
                    try {
                        const refinedResult = await textSummarizer.summarize(
                            visionResult.analysis,
                            {
                                ...summarySettings,
                                mode: 'refine',
                                category: (target as any).category || 'Misc',
                                type: target.type
                            } as any
                        )

                        if (refinedResult?.summary) {
                            finalAnalysis = refinedResult.summary
                        }
                    } catch (err) {
                        console.error("Refinement failed", err)
                    }
                }

                setDetectionResult(prev => {
                    if (!prev) return prev
                    return {
                        ...prev,
                        data: prev.data.map(d =>
                            (d.x === target.x && d.y === target.y && d.type === target.type)
                                ? { ...d, analysis: finalAnalysis, model: analysisModel }
                                : d
                        )
                    }
                })
            }))
        } finally {
            setTimeout(() => {
                analyzingRef.current.delete(analyzeKey)
            }, 1000)
        }
    }, [deepAnalysisThreshold, categoryThresholds, summarySettings, enableEnhancedDescription, visionModel, lmStudioUrl])

    const performScan = useCallback(async (video: HTMLVideoElement) => {
        console.log(`[VideoScanner performScan] Starting, enableDeepAnalysis=${enableDeepAnalysis}, visionModel=${visionModel}`)
        setIsScanning(true)
        setDetectionResult(null)

        const frameData = await imageScanner.captureElementScreenshot(video, saveScannedImages);
        if (!frameData) {
            setIsScanning(false)
            return
        }

        const res = await imageScanner.detectImageData(frameData, saveScannedImages, visionModel, visionModel, lmStudioUrl, true)
        console.log(`[VideoScanner performScan] Got ${res?.data?.length || 0} objects`)

        // Only update if we are still hovering the same video
        if (video === hoveredVideoRef.current) {
            setDetectionResult(res)
            setIsScanning(false)
            // Cache the result with analysis
            if (res) {
                const cacheKey = `video-${video.src || video.currentSrc}`
                detectionCacheRef.current.set(cacheKey, res)
                // Also persist to localStorage
                try {
                    localStorage.setItem(`ai-scanner-cache-${cacheKey}`, JSON.stringify({
                        timestamp: Date.now(),
                        data: res.data
                    }))
                } catch (e) { /* ignore storage errors */ }
            }

            if (res && enableDeepAnalysis) {
                console.log(`[VideoScanner performScan] Triggering analysis`)
                triggerDeepAnalysis(res)
            } else {
                console.log(`[VideoScanner performScan] Skipping analysis: enableDeepAnalysis=${enableDeepAnalysis}`)
            }
        }
    }, [saveScannedImages, enableDeepAnalysis, triggerDeepAnalysis, visionModel, lmStudioUrl])

    useEffect(() => {
        if (!isActive) {
            setHoveredVideo(null)
            hoveredVideoRef.current = null
            setDetectionResult(null)
            setIsScanning(false)
            if (clearTimeoutRef.current) clearTimeout(clearTimeoutRef.current)
            return
        }

        const handleMouseMove = (e: MouseEvent) => {
            setMousePos({ x: e.clientX, y: e.clientY })
            const video = findVideoUnderCursor(e)

            // Case 1: We found a paused video
            if (video && video.paused) {
                // Is it a new paused video?
                if (video === hoveredVideoRef.current) {
                    // Same video, do nothing (already scanned or cached)
                } else {
                    hoveredVideoRef.current = video
                    setHoveredVideo(video)
                    // Check cache first before scanning
                    const cacheKey = `video-${video.src || video.currentSrc}`
                    const cached = detectionCacheRef.current.get(cacheKey) || (() => {
                        try {
                            const stored = localStorage.getItem(`ai-scanner-cache-${cacheKey}`)
                            if (stored) {
                                const parsed = JSON.parse(stored)
                                // Check if cache is fresh (less than 5 minutes old)
                                if (Date.now() - parsed.timestamp < 5 * 60 * 1000) {
                                    return { data: parsed.data } as DetectionResult
                                }
                            }
                        } catch (e) { /* ignore */ }
                        return null
                    })()
                    if (cached && cached.data?.length > 0) {
                        console.log(`[useVideoScanner] Restoring from cache`)
                        setDetectionResult(cached)
                    } else {
                        performScan(video) // Scan immediately
                    }
                }
                // If it's the same paused video, do nothing. We've already scanned it.
            }
            // Case 2: No video, or a playing video
            else {
                // Did we *just* move off a video?
                // Cancel any pending clear
                if (clearTimeoutRef.current) {
                    clearTimeout(clearTimeoutRef.current)
                    clearTimeoutRef.current = null
                }

                // Don't clear if hovering the sidebar (for scrolling)
                if (hoveredVideoRef.current && !isHoveringSidebar) {
                    // Delay clearing to allow time to travel through bridge to sidebar
                    clearTimeoutRef.current = setTimeout(() => {
                        // Double-check we're still not hovering sidebar before clearing
                        if (!isHoveringSidebar) {
                            hoveredVideoRef.current = null
                            setHoveredVideo(null)
                            setDetectionResult(null)
                            setIsScanning(false)
                        }
                    }, 400) // 400ms grace period to reach sidebar
                }
            }
        }

        window.addEventListener('mousemove', handleMouseMove)
        return () => {
            window.removeEventListener('mousemove', handleMouseMove)
            if (clearTimeoutRef.current) clearTimeout(clearTimeoutRef.current)
        }
    }, [isActive, findVideoUnderCursor, performScan, isHoveringSidebar])

    return {
        hoveredVideo,
        detectionResult,
        isScanning,
        mousePos,
        // Expose a manual trigger for retrying or refreshing
        rescan: () => {
            if (hoveredVideo) {
                performScan(hoveredVideo)
            }
        }
    }
}
