import React, { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Scan, Loader2, AlertCircle, Bot } from 'lucide-react'
import type { DetectionResult } from '../services/imageScanner'

interface DetectionOverlayProps {
    detectionResult: DetectionResult | null
    isScanning: boolean
    mousePos: { x: number; y: number }
    targetElement: HTMLElement | null
}

const DetectionOverlay: React.FC<DetectionOverlayProps> = ({
    detectionResult,
    isScanning,
    mousePos,
    targetElement
}) => {
    const [position, setPosition] = useState({ left: 0, top: 0 })

    useEffect(() => {
        let left = mousePos.x + 20
        let top = mousePos.y

        if (targetElement) {
            const rect = targetElement.getBoundingClientRect()
            left = rect.right + 16
            top = rect.top
        }

        // Clamp to viewport
        const maxLeft = window.innerWidth - 400 
        const maxTop = window.innerHeight - 300

        const clampedLeft = Math.max(16, Math.min(left, maxLeft))
        const clampedTop = Math.max(16, Math.min(top, maxTop))

        setPosition({ left: clampedLeft, top: clampedTop })
    }, [targetElement, mousePos])
    
    const hasDetections = detectionResult && detectionResult.data.length > 0
    const shouldShow = isScanning || hasDetections

    if (!shouldShow) return null

    return (
        <AnimatePresence>
            <motion.div
                initial={{ opacity: 0, scale: 0.9, y: -15 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.9, y: -15 }}
                transition={{ duration: 0.25, ease: 'easeOut' }}
                className="fixed pointer-events-none z-[100000]"
                style={{
                    left: `${position.left}px`,
                    top: `${position.top}px`,
                }}
            >
                <div className="relative">
                    <div className="absolute inset-0 bg-cyan-500/10 blur-lg rounded-lg" />
                    <div className="relative bg-slate-900/90 backdrop-blur-md border border-cyan-500/30 rounded-lg shadow-2xl max-w-sm w-full">
                        <div className="absolute -top-1.5 -left-1.5 w-4 h-4 border-t-2 border-l-2 border-cyan-400/80" />
                        <div className="absolute -top-1.5 -right-1.5 w-4 h-4 border-t-2 border-r-2 border-cyan-400/80" />
                        <div className="absolute -bottom-1.5 -left-1.5 w-4 h-4 border-b-2 border-l-2 border-cyan-400/80" />
                        <div className="absolute -bottom-1.5 -right-1.5 w-4 h-4 border-b-2 border-r-2 border-cyan-400/80" />

                        <div className="p-4 space-y-3">
                            <div className="flex items-center gap-3 border-b border-cyan-500/20 pb-2">
                                <Scan className="w-5 h-5 text-cyan-400" />
                                <h3 className="text-md font-bold text-cyan-300 tracking-wider uppercase">
                                    AI Object Detection
                                </h3>
                            </div>

                            <div className="min-h-[80px] max-h-[400px] overflow-y-auto pointer-events-auto scrollbar-thin scrollbar-thumb-cyan-700/60 scrollbar-track-slate-800/50 pr-2">
                                {isScanning && !hasDetections && (
                                    <div className="flex items-center justify-center h-24 gap-2 text-cyan-300">
                                        <Loader2 className="w-5 h-5 animate-spin" />
                                        <span className="text-sm animate-pulse">Scanning frame...</span>
                                    </div>
                                )}
                                {hasDetections ? (
                                    <ul className="space-y-3">
                                        {detectionResult.data.map((item, index) => (
                                            <li key={index} className="flex gap-3 items-start text-sm">
                                                <div className="w-16 flex-shrink-0 text-right">
                                                    <span className="font-mono text-xs text-cyan-400 bg-cyan-900/50 px-2 py-0.5 rounded">
                                                        {item.type}
                                                    </span>
                                                </div>
                                                <div className="flex-1 text-slate-300">
                                                    {item.analysis ? (
                                                         item.analysis === '...' ? (
                                                            <div className="flex items-center gap-2 text-yellow-400 text-xs">
                                                                <Bot className="w-4 h-4 animate-pulse" />
                                                                <span>Deep analysis in progress...</span>
                                                            </div>
                                                         ) : (
                                                            <p>{item.analysis}</p>
                                                         )
                                                    ) : (
                                                        <span className="text-slate-500 italic">Standard detection.</span>
                                                    )}
                                                </div>
                                            </li>
                                        ))}
                                    </ul>
                                ) : !isScanning && (
                                     <div className="flex items-center justify-center h-24 gap-2 text-slate-500">
                                        <AlertCircle className="w-5 h-5" />
                                        <span className="text-sm">No objects detected.</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </motion.div>
        </AnimatePresence>
    )
}

export default DetectionOverlay
