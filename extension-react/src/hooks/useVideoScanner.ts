import { useState, useEffect, useCallback, useRef } from 'react'
import { imageScanner, type DetectionResult } from '../services/imageScanner'
import { type SummarySettings, textSummarizer } from '../services/textSummarizer'

export function useVideoScanner(
    isActive: boolean,
    saveScannedImages: boolean = false,
    enableDeepAnalysis: boolean = false,
    enableEnhancedDescription: boolean = true,
    deepAnalysisThreshold: number = 0.85,
    categoryThresholds: Record<string, number> = {},
    summarySettings?: SummarySettings
) {
    const [hoveredVideo, setHoveredVideo] = useState<HTMLVideoElement | null>(null)
    const hoveredVideoRef = useRef<HTMLVideoElement | null>(null)
    const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null)
    const [isScanning, setIsScanning] = useState(false)
    const [mousePos, setMousePos] = useState({ x: 0, y: 0 })
    const analyzingRef = useRef<Set<string>>(new Set())

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
            return isAnalyzable && d.confidence >= threshold && !d.analysis
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
                    return (isAnalyzable && d.confidence >= threshold && !d.analysis) ? { ...d, analysis: '...' } : d
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
                    target.type
                )

                if (!visionResult) return

                let finalAnalysis = visionResult

                const isAnalyzable = (target as any).is_analyzable
                if (enableEnhancedDescription && isAnalyzable && visionResult && !visionResult.startsWith('Error') && summarySettings) {
                    try {
                        const refinedResult = await textSummarizer.summarize(
                            visionResult,
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
                                ? { ...d, analysis: finalAnalysis }
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
    }, [deepAnalysisThreshold, categoryThresholds, summarySettings, enableEnhancedDescription])

    const performScan = useCallback(async (video: HTMLVideoElement) => {
        setIsScanning(true)
        setDetectionResult(null)

        const frameData = await imageScanner.captureElementScreenshot(video, saveScannedImages);
        if (!frameData) {
            setIsScanning(false)
            return
        }

        const res = await imageScanner.detectImageData(frameData, saveScannedImages)
        
        // Only update if we are still hovering the same video
        if (video === hoveredVideoRef.current) {
            setDetectionResult(res)
            setIsScanning(false)

            if (res && enableDeepAnalysis) {
                triggerDeepAnalysis(res)
            }
        }
    }, [saveScannedImages, enableDeepAnalysis, triggerDeepAnalysis])

    useEffect(() => {
        if (!isActive) {
            setHoveredVideo(null)
            hoveredVideoRef.current = null
            setDetectionResult(null)
            setIsScanning(false)
            return
        }

        const handleMouseMove = (e: MouseEvent) => {
            setMousePos({ x: e.clientX, y: e.clientY })
            const video = findVideoUnderCursor(e)

            // Case 1: We found a paused video
            if (video && video.paused) {
                // Is it a new paused video?
                if (video !== hoveredVideoRef.current) {
                    hoveredVideoRef.current = video
                    setHoveredVideo(video)
                    performScan(video) // Scan immediately
                }
                // If it's the same paused video, do nothing. We've already scanned it.
            } 
            // Case 2: No video, or a playing video
            else {
                // Did we *just* move off a video?
                if (hoveredVideoRef.current) {
                    hoveredVideoRef.current = null
                    setHoveredVideo(null)
                    setDetectionResult(null)
                    setIsScanning(false)
                }
            }
        }

        window.addEventListener('mousemove', handleMouseMove)
        return () => {
            window.removeEventListener('mousemove', handleMouseMove)
        }
    }, [isActive, findVideoUnderCursor, performScan])

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
