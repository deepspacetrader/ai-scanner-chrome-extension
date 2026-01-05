import { useState, useEffect, useCallback, useRef } from 'react'
import { imageScanner, type DetectionResult } from '../services/imageScanner'
import { type SummarySettings, textSummarizer } from '../services/textSummarizer'

export function useScanner(
    isActive: boolean,
    saveScannedImages: boolean = false,
    enableDeepAnalysis: boolean = false,
    enableEnhancedDescription: boolean = true,
    deepAnalysisThreshold: number = 0.85,
    categoryThresholds: Record<string, number> = {},
    summarySettings?: SummarySettings,
    visionModel: string = 'florence2'
) {
    const [hoveredElement, setHoveredElement] = useState<HTMLImageElement | HTMLVideoElement | null>(null)
    const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null)
    const [isScanning, setIsScanning] = useState(false)
    const [mousePos, setMousePos] = useState({ x: 0, y: 0 })
    const analyzingRef = useRef<Set<string>>(new Set())
    const scanTimeoutRef = useRef<NodeJS.Timeout | null>(null)
    const canvasRef = useRef<HTMLCanvasElement | null>(null)

    const findElementUnderCursor = useCallback((e: MouseEvent) => {
        const elements = document.elementsFromPoint(e.clientX, e.clientY)
        for (const el of elements) {
            if (el instanceof HTMLImageElement) {
                return el
            }
            if (el instanceof HTMLVideoElement) {
                // Additional check for YouTube to avoid triggering on controls
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

    const captureFrame = useCallback((video: HTMLVideoElement): string | null => {
        if (!canvasRef.current) {
            canvasRef.current = document.createElement('canvas')
        }
        const canvas = canvasRef.current
        // Check if video has metadata loaded
        if (video.videoWidth === 0 || video.videoHeight === 0) {
            return null
        }
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        const ctx = canvas.getContext('2d')
        if (!ctx) return null

        try {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
            return canvas.toDataURL('image/jpeg', 0.9) // Use JPEG for smaller size
        } catch (error) {
            console.error('Error capturing video frame:', error)
            return null
        }
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
                    target.type,
                    visionModel
                ) as any
                if (!visionResult) return

                let finalAnalysis = visionResult.analysis || visionResult
                let analysisModel = visionResult.model || (visionModel === 'glm4.6v' ? 'Qwen2-VL' : 'Florence-2')

                const isAnalyzable = (target as any).is_analyzable
                if (enableEnhancedDescription && isAnalyzable && visionResult && !visionResult.startsWith('Error') && summarySettings) {
                    try {
                        const refinedResult = await textSummarizer.summarize(visionResult, {
                            ...summarySettings,
                            mode: 'refine',
                            category: (target as any).category || 'Misc',
                            type: target.type
                        } as any)
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
    }, [deepAnalysisThreshold, categoryThresholds, summarySettings, enableEnhancedDescription, visionModel])

    const performVideoScan = useCallback(async (video: HTMLVideoElement) => {
        setIsScanning(true)
        setDetectionResult(null)

        const frameData = captureFrame(video)
        if (!frameData) {
            setIsScanning(false)
            return
        }

        const res = await imageScanner.detectImageData(frameData, saveScannedImages)

        if (hoveredElement === video) {
            setDetectionResult(res)
            setIsScanning(false)
            if (res && enableDeepAnalysis) {
                triggerDeepAnalysis(res)
            }
        }
    }, [captureFrame, saveScannedImages, enableDeepAnalysis, triggerDeepAnalysis, hoveredElement])

    useEffect(() => {
        if (!isActive) {
            setHoveredElement(null)
            setDetectionResult(null)
            if (scanTimeoutRef.current) clearTimeout(scanTimeoutRef.current)
            return
        }

        const handleMouseMove = async (e: MouseEvent) => {
            setMousePos({ x: e.clientX, y: e.clientY })

            const el = findElementUnderCursor(e)

            if (el !== hoveredElement) {
                setHoveredElement(el)
                setDetectionResult(null)
                setIsScanning(false)
                if (scanTimeoutRef.current) clearTimeout(scanTimeoutRef.current)

                if (el instanceof HTMLImageElement) {
                    const src = imageScanner.getImageSourceKey(el)
                    const cached = imageScanner.getCachedDetection(src)
                    if (cached) {
                        setDetectionResult(cached)
                        if (enableDeepAnalysis && cached.data.some(d => (d as any).is_analyzable && d.confidence >= (categoryThresholds[(d as any).category || 'Misc'] ?? deepAnalysisThreshold) && !d.analysis)) {
                            triggerDeepAnalysis(cached)
                        }
                    } else {
                        setIsScanning(true)
                        const res = await imageScanner.detectImage(el, saveScannedImages)
                        if (el === findElementUnderCursor(e)) {
                            setDetectionResult(res)
                            setIsScanning(false)
                            if (res && enableDeepAnalysis) {
                                triggerDeepAnalysis(res)
                            }
                        }
                    }
                } else if (el instanceof HTMLVideoElement) {
                    scanTimeoutRef.current = setTimeout(() => {
                        performVideoScan(el)
                    }, 200)
                }
            }
        }

        window.addEventListener('mousemove', handleMouseMove)
        return () => {
            window.removeEventListener('mousemove', handleMouseMove)
            if (scanTimeoutRef.current) clearTimeout(scanTimeoutRef.current)
        }
    }, [isActive, hoveredElement, findElementUnderCursor, performVideoScan, saveScannedImages, enableDeepAnalysis, categoryThresholds, deepAnalysisThreshold, visionModel])

    const rescan = useCallback(() => {
        if (hoveredElement instanceof HTMLVideoElement) {
            performVideoScan(hoveredElement)
        }
        // Rescan for images can be added here if needed
    }, [hoveredElement, performVideoScan])

    return {
        hoveredElement,
        detectionResult,
        isScanning,
        mousePos,
        rescan
    }
}
