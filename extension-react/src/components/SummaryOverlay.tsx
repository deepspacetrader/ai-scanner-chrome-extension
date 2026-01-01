import React, { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { FileText, Loader2, AlertCircle } from 'lucide-react'
import type { TextSelection } from '../hooks/useTextSummarization'
import type { SummaryResult } from '../services/textSummarizer'

interface SummaryOverlayProps {
    selection: TextSelection | null
    summaryResult: SummaryResult | null
    isSummarizing: boolean
    error: string
    mousePos: { x: number; y: number }
}

const SummaryOverlay: React.FC<SummaryOverlayProps> = ({
    selection,
    summaryResult,
    isSummarizing,
    error,
    mousePos,
}) => {
    const [position, setPosition] = useState({ left: 0, top: 0 })

    useEffect(() => {
        if (!selection) return

        let left = mousePos.x + 16
        let top = mousePos.y - 20

        // If we have a selection rect, position relative to it
        if (selection.rect) {
            left = selection.rect.left + selection.rect.width + 16
            top = selection.rect.top - 10
        }

        // Clamp to viewport
        const maxLeft = window.innerWidth - 380
        const maxTop = window.innerHeight - 200

        const clampedLeft = Math.max(8, Math.min(left, maxLeft))
        const clampedTop = Math.max(8, Math.min(top, maxTop))

        setPosition({ left: clampedLeft, top: clampedTop })
    }, [selection, mousePos])

    if (!selection) return null

    const isFullPage = selection.rawText.length > 5000

    return (
        <AnimatePresence>
            <motion.div
                initial={{ opacity: 0, scale: 0.95, y: -10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95, y: -10 }}
                transition={{ duration: 0.2, ease: 'easeOut' }}
                className="fixed pointer-events-none z-[100000]"
                style={{
                    left: `${position.left}px`,
                    top: `${position.top}px`,
                }}
            >
                <div className="relative">
                    {/* Cyberpunk glow effect */}
                    <div className="absolute inset-0 bg-cyan-500/20 blur-xl rounded-lg" />

                    {/* Main container */}
                    <div className="relative bg-[#0d1117] border border-cyan-500/50 rounded-lg shadow-[0_0_30px_rgba(0,0,0,0.6)] max-w-[360px] min-w-[260px]">
                        {/* Animated border effect */}
                        <div className="absolute inset-0 rounded-lg overflow-hidden">
                            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/0 via-cyan-500/10 to-cyan-500/0 animate-pulse" style={{ background: '#00131b' }} />
                        </div>

                        {/* Content */}
                        <div className="relative p-4 space-y-3">
                            {/* Header */}
                            <div className="flex items-center gap-2 border-b border-cyan-500/30 pb-2">
                                <FileText className="w-4 h-4 text-cyan-400" />
                                <span className="text-sm font-semibold text-cyan-300 uppercase tracking-wider">
                                    {isFullPage ? 'Full Page Summary' : 'Text Summary'}
                                </span>
                            </div>

                            {/* Preview
                            {selection.preview && (
                                <div className="text-xs text-slate-300 leading-relaxed line-clamp-3 font-mono">
                                    {selection.preview}
                                </div>
                            )} */}

                            {/* Summary content - scrollable */}
                            <div className="min-h-[60px] max-h-[200px] overflow-y-auto pointer-events-auto scrollbar-thin scrollbar-thumb-cyan-500/50 scrollbar-track-transparent" style={{ borderTop: '1px solid rgb(103, 232, 249)', scrollbarWidth: 'thin', scrollbarColor: 'rgb(103, 232, 249) transparent', marginTop: '0' }}>
                                {error ? (
                                    <div className="flex items-start gap-2 text-red-400 text-sm">
                                        <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                                        <span>{error}</span>
                                    </div>
                                ) : summaryResult?.summary ? (
                                    <div className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
                                        {summaryResult.summary}
                                    </div>
                                ) : isSummarizing ? (
                                    <div className="flex items-center gap-2 text-cyan-300 text-sm">
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        <span className="animate-pulse">Analyzing patterns…</span>
                                    </div>
                                ) : (
                                    <div className="text-sm text-slate-300 italic">
                                        Highlight text and hold the trigger key to summarize.
                                    </div>
                                )}
                            </div>

                            {/* Footer metadata */}
                            {summaryResult?.model && (
                                <div className="flex items-center gap-2 pt-2 border-t border-cyan-500/20">
                                    <div className="flex-1 text-[10px] uppercase tracking-widest text-slate-500 font-mono">
                                        Local · {summaryResult.model}
                                    </div>
                                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.8)]" />
                                </div>
                            )}

                            {/* Cyberpunk corner accents */}
                            <div className="absolute -top-1 -left-1 w-3 h-3 border-t-2 border-l-2 border-cyan-400" />
                            <div className="absolute -top-1 -right-1 w-3 h-3 border-t-2 border-r-2 border-cyan-400" />
                            <div className="absolute -bottom-1 -left-1 w-3 h-3 border-b-2 border-l-2 border-cyan-400" />
                            <div className="absolute -bottom-1 -right-1 w-3 h-3 border-b-2 border-r-2 border-cyan-400" />
                        </div>

                        {/* Edge scan effect - stays at borders to avoid covering text */}
                        {!isFullPage && (
                            <div className="absolute inset-0 rounded-lg overflow-hidden pointer-events-none">
                                <motion.div
                                    className="absolute top-0 left-0 h-[1.5px] bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.6)]"
                                    animate={{ left: ['-20%', '120%'] }}
                                    transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
                                    style={{ width: '25%' }}
                                />
                                <motion.div
                                    className="absolute bottom-0 right-0 h-[1.5px] bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.6)]"
                                    animate={{ right: ['-20%', '120%'] }}
                                    transition={{ duration: 3, repeat: Infinity, ease: 'linear', delay: 1.5 }}
                                    style={{ width: '25%' }}
                                />
                                <motion.div
                                    className="absolute left-0 top-0 w-[1.5px] bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.6)]"
                                    animate={{ top: ['-20%', '120%'] }}
                                    transition={{ duration: 3, repeat: Infinity, ease: 'linear', delay: 0.75 }}
                                    style={{ height: '25%' }}
                                />
                                <motion.div
                                    className="absolute right-0 bottom-0 w-[1.5px] bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.6)]"
                                    animate={{ bottom: ['-20%', '120%'] }}
                                    transition={{ duration: 3, repeat: Infinity, ease: 'linear', delay: 2.25 }}
                                    style={{ height: '25%' }}
                                />
                            </div>
                        )}
                    </div>
                </div>
            </motion.div>
        </AnimatePresence>
    )
}

export default SummaryOverlay
