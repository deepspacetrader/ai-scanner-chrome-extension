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
    summarySettings?: SummarySettings
) {
    const [hoveredImage, setHoveredImage] = useState<HTMLImageElement | null>(null)
    const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null)
    const [isScanning, setIsScanning] = useState(false)
    const [mousePos, setMousePos] = useState({ x: 0, y: 0 })
    const analyzingRef = useRef<Set<string>>(new Set())

    // Find image under cursor
    const findImageUnderCursor = useCallback((e: MouseEvent) => {
        // Use elementsFromPoint to find elements at the cursor even if covered by transparent overlays
        const elements = document.elementsFromPoint(e.clientX, e.clientY);

        for (const el of elements) {
            // Check if element is an image
            if (el instanceof HTMLImageElement) return el;

            // Optional: Check if we are hovering a container that wraps the image?
            // Usually finding the image directly is safer with elementsFromPoint.
            // But if the structure is specifically:
            // <div> <div overlay></div> <img /> </div>
            // The <img> will be in the elements list if it's visually at that point.
        }

        return null;
    }, [])

    const triggerDeepAnalysis = useCallback(async (result: DetectionResult) => {
        if (!result.image) return

        // Use per-category thresholds if available
        const targets = result.data.filter(d => {
            const category = (d as any).category || "Misc"
            const threshold = categoryThresholds[category] ?? deepAnalysisThreshold
            const isAnalyzable = (d as any).is_analyzable
            const meetThreshold = d.confidence >= threshold

            if (isAnalyzable && !d.analysis) {
                console.log(`[Scanner] Checking ${d.type} (${category}): conf=${d.confidence}, threshold=${threshold}, meet=${meetThreshold}`)
            }

            return isAnalyzable && meetThreshold && !d.analysis
        })

        if (targets.length === 0) {
            console.log(`[Scanner] No targets found for deep analysis. Total objects: ${result.data.length}`)
            return
        }

        console.log(`[Scanner] Triggering deep analysis for ${targets.length} objects`)

        // Use a stable key
        const analyzeKey = `${result.image.substring(0, 50)}_${targets.length}`
        if (analyzingRef.current.has(analyzeKey)) return
        analyzingRef.current.add(analyzeKey)

        // Set initial loading state
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
                // Step 1: Get Florence-2 Analysis (The "Vision" step)
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

                // Step 2: Use the unified summarizer to refine the output (The "Brain" step)
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
    }, [deepAnalysisThreshold, categoryThresholds, summarySettings])

    useEffect(() => {
        if (!isActive) {
            setHoveredImage(null)
            setDetectionResult(null)
            return
        }

        const handleMouseMove = async (e: MouseEvent) => {
            setMousePos({ x: e.clientX, y: e.clientY })

            const img = findImageUnderCursor(e)

            if (img !== hoveredImage) {
                setHoveredImage(img)
                if (img) {
                    // Check cache first
                    const src = imageScanner.getImageSourceKey(img)
                    const cached = imageScanner.getCachedDetection(src)
                    if (cached) {
                        setDetectionResult(cached)
                        setIsScanning(false)

                        // If cached has no analysis but deep analysis is now enabled, we might want to re-trigger?
                        // For now, let's just use cache as is.
                        const hasAnalysisPending = cached.data.some(d => {
                            const category = (d as any).category || "Misc"
                            const threshold = categoryThresholds[category] ?? deepAnalysisThreshold
                            const isAnalyzable = (d as any).is_analyzable
                            return isAnalyzable && d.confidence >= threshold && !d.analysis
                        })

                        if (enableDeepAnalysis && hasAnalysisPending) {
                            triggerDeepAnalysis(cached)
                        }
                    } else {
                        setDetectionResult(null)
                        setIsScanning(true)
                        const res = await imageScanner.detectImage(img, saveScannedImages)
                        // Verify we are still hovering the same image
                        if (img === findImageUnderCursor(e)) {
                            setDetectionResult(res)
                            setIsScanning(false)

                            if (res && enableDeepAnalysis) {
                                triggerDeepAnalysis(res)
                            }
                        }
                    }
                } else {
                    setDetectionResult(null)
                    setIsScanning(false)
                }
            }
        }

        window.addEventListener('mousemove', handleMouseMove)
        return () => window.removeEventListener('mousemove', handleMouseMove)
    }, [isActive, hoveredImage, findImageUnderCursor, categoryThresholds, deepAnalysisThreshold, enableDeepAnalysis, saveScannedImages])

    return {
        hoveredImage,
        detectionResult,
        isScanning,
        mousePos
    }
}
