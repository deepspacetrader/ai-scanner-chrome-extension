import React, { useState, useEffect } from 'react'
import { Brain, X, Target, Loader2, CheckCircle } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { useScanner } from '../hooks/useScanner'
import { useTextSummarization } from '../hooks/useTextSummarization'
import { useVideoScanner } from '../hooks/useVideoScanner'
import SummaryOverlay from './SummaryOverlay'
import type { SummarySettings } from '../services/textSummarizer'
import { audioService } from '../services/audioService'
import type { Settings } from '../types/settings'
import { DEFAULT_SETTINGS } from '../types/settings'

interface ScannerHUDProps {
    shadowRoot?: ShadowRoot
}

const ScannerHUD: React.FC<ScannerHUDProps> = () => {
    const [isActive, setIsActive] = useState(false)
    const [isMouseDown, setIsMouseDown] = useState(false)
    const [isHoveringText, setIsHoveringText] = useState(false)
    const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS)
    const summarySettings: SummarySettings = {
        endpoint: settings.summarizationEndpoint,
        model: settings.summarizationModel,
        minChars: settings.minSummaryChars,
    }

    // Scanner for images
    const {
        hoveredElement: imageHoveredElement,
        detectionResult: imageDetectionResult,
        isScanning: isImageScanning,
        mousePos
    } = useScanner(
        isActive,
        settings.saveScannedImages,
        settings.enableDeepAnalysis,
        settings.enableEnhancedDescription,
        settings.deepAnalysisThreshold,
        settings.categoryThresholds,
        summarySettings
    )

    // Scanner for videos
    const {
        hoveredVideo,
        detectionResult: videoDetectionResult,
        isScanning: isVideoScanning,
        mousePos: videoMousePos
    } = useVideoScanner(
        isActive,
        settings.saveScannedImages,
        settings.enableDeepAnalysis,
        settings.enableEnhancedDescription,
        settings.deepAnalysisThreshold,
        settings.categoryThresholds,
        summarySettings
    );

    const hoveredImage = imageHoveredElement instanceof HTMLImageElement ? imageHoveredElement : null;
    const hoveredElement = hoveredImage || hoveredVideo;
    const detectionResult = hoveredImage ? imageDetectionResult : videoDetectionResult;
    const isScanning = isImageScanning || isVideoScanning;
    const activeMousePos = hoveredImage ? mousePos : videoMousePos;

    const {
        selection,
        summaryResult,
        isSummarizing,
        error: summaryError,
    } = useTextSummarization(isActive && settings.enableSummarization, summarySettings)

    // Audio Triggers
    useEffect(() => {
        if (isScanning && settings.enableSound) {
            audioService.playScan(settings.soundVolume)
        }
    }, [isScanning, settings.enableSound, settings.soundVolume])

    useEffect(() => {
        if (detectionResult && detectionResult.data.length > 0 && settings.enableSound) {
            audioService.playLock(settings.soundVolume)
        }
    }, [detectionResult, settings.enableSound, settings.soundVolume])

    const getSystemStatus = () => {
        if (isScanning) return 'SCANNING';
        if (detectionResult) {
            const isProcessingFlorence = detectionResult.data.some(d => d.analysis === '...');
            if (isProcessingFlorence) {
                return 'PROCESSING';
            }
            return 'DONE';
        }
        if (isSummarizing) return 'SUMMARIZING';
        if (isMouseDown && !hoveredElement) return 'SELECTING';
        if (isHoveringText) return 'TEXT';
        if (hoveredElement) return 'READY';
        return 'IDLE';
    };

    const systemStatus = getSystemStatus()

    // Load settings from chrome storage and listen for changes
    useEffect(() => {
        const loadSettings = (callback?: () => void) => {
            if (chrome?.storage?.sync) {
                chrome.storage.sync.get(Object.keys(DEFAULT_SETTINGS), (result) => {
                    const mergedSettings = { ...DEFAULT_SETTINGS, ...result }
                    if (typeof mergedSettings.minSummaryChars !== 'number') {
                        mergedSettings.minSummaryChars = DEFAULT_SETTINGS.minSummaryChars
                    }

                    // Auto-migrate from Granite to Qwen if old model is detected
                    if (mergedSettings.summarizationModel.toLowerCase().includes('granite')) {
                        mergedSettings.summarizationModel = DEFAULT_SETTINGS.summarizationModel;
                    }

                    setSettings(mergedSettings as Settings)
                    callback?.()
                })
            }
        }

        loadSettings()

        // Listen for storage changes (when user updates settings from popup)
        const handleStorageChange = () => {
            loadSettings()
        }

        chrome?.storage?.onChanged?.addListener(handleStorageChange)
        return () => {
            chrome?.storage?.onChanged?.removeListener(handleStorageChange)
        }
    }, [])

    // Use ref to avoid stale closure issues in event handlers
    const settingsRef = React.useRef(settings)
    React.useEffect(() => {
        settingsRef.current = settings
    }, [settings])

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.repeat) return

            const currentSettings = settingsRef.current
            const [inputType, inputValue] = currentSettings.triggerInput.split(':')

            if (inputType === 'keyboard' && (e.key === inputValue || e.code === inputValue)) {
                if (currentSettings.toggleActivation) {
                    setIsActive(prev => !prev)
                } else {
                    setIsActive(true)
                }
                e.preventDefault()
                e.stopPropagation()
            }

            if (e.key === 'Escape') {
                setIsActive(false)
            }
        }

        const handleKeyUp = (e: KeyboardEvent) => {
            const currentSettings = settingsRef.current
            if (currentSettings.toggleActivation) return

            const [inputType, inputValue] = currentSettings.triggerInput.split(':')

            if (inputType === 'keyboard' && (e.key === inputValue || e.code === inputValue)) {
                setIsActive(false)
                e.preventDefault()
                e.stopPropagation()
            }
        }

        const handleMouseDown = (e: MouseEvent) => {
            const currentSettings = settingsRef.current
            const [inputType, inputValue] = currentSettings.triggerInput.split(':')

            if (inputType === 'mouse' && e.button === parseInt(inputValue, 10)) {
                if (currentSettings.toggleActivation) {
                    setIsActive(prev => !prev)
                } else {
                    setIsActive(true)
                }
                e.preventDefault()
                e.stopPropagation()
            }
        }

        const handleMouseUp = (e: MouseEvent) => {
            const currentSettings = settingsRef.current
            if (currentSettings.toggleActivation) return

            const [inputType, inputValue] = currentSettings.triggerInput.split(':')

            if (inputType === 'mouse' && e.button === parseInt(inputValue, 10)) {
                setIsActive(false)
                e.preventDefault()
                e.stopPropagation()
            }
        }

        window.addEventListener('keydown', handleKeyDown)
        window.addEventListener('keyup', handleKeyUp)
        window.addEventListener('mousedown', handleMouseDown)
        window.addEventListener('mouseup', handleMouseUp)

        return () => {
            window.removeEventListener('keydown', handleKeyDown)
            window.removeEventListener('keyup', handleKeyUp)
            window.removeEventListener('mousedown', handleMouseDown)
            window.removeEventListener('mouseup', handleMouseUp)
        }
    }, [])

    useEffect(() => {
        const onMouseDown = () => setIsMouseDown(true)
        const onMouseUp = () => setIsMouseDown(false)
        const onMouseMove = (e: MouseEvent) => {
            const el = document.elementFromPoint(e.clientX, e.clientY)
            if (el) {
                const style = window.getComputedStyle(el)
                setIsHoveringText(style.cursor === 'text')
            } else {
                setIsHoveringText(false)
            }
        }

        window.addEventListener('mousedown', onMouseDown)
        window.addEventListener('mouseup', onMouseUp)
        window.addEventListener('mousemove', onMouseMove)
        return () => {
            window.removeEventListener('mousedown', onMouseDown)
            window.removeEventListener('mouseup', onMouseUp)
            window.removeEventListener('mousemove', onMouseMove)
        }
    }, [])

    if (!isActive) return null

    const renderSystemStatus = () => {
        switch (systemStatus) {
            case 'SCANNING':
                return (
                    <div className="flex items-center gap-2 text-yellow-400">
                        <Loader2 className="w-3 h-3 animate-spin" />
                        <span>ANALYZING FRAME</span>
                    </div>
                );
            case 'PROCESSING':
                return (
                    <div className="flex items-center gap-2 text-yellow-400">
                        <Loader2 className="w-3 h-3 animate-spin" />
                        <span>PROCESSING</span>
                    </div>
                );
            case 'DONE':
                return (
                    <div className="flex items-center gap-2 text-green-400">
                        <CheckCircle className="w-3 h-3" />
                        <span>DONE</span>
                    </div>
                );
            default:
                return (
                    <div className="flex items-center gap-2">
                        <div className={`w-1.5 h-1.5 rounded-full ${systemStatus !== 'IDLE' ? 'bg-cyan-400 animate-ping-pong' : 'bg-gray-500'}`} />
                        <span>SYSTEM: {systemStatus}</span>
                    </div>
                );
        }
    }

    return (
        <div
            id="ai-scanner-hud-container"
            className="fixed inset-0 pointer-events-none z-[99999] font-mono text-cyan-400"
        >
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_0%,rgba(0,0,0,0.25)_100%)] pointer-events-none" />
            <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-10 pointer-events-none mix-blend-overlay" />
            <div className="absolute inset-0 border-[20px] border-cyan-900/20 box-border pointer-events-none" />

            <div className="absolute top-0 left-0 right-0 h-16 bg-gradient-to-b from-black/80 to-transparent flex items-center px-8 border-b border-cyan-500/30 backdrop-blur-sm pointer-events-auto">
                <div className="flex items-center space-x-4">
                    <Brain className="w-6 h-6 animate-pulse" />
                    <span className="text-xl font-bold tracking-widest text-shadow-cyber">AI SCANNER: ONLINE</span>
                </div>
                <div className="flex-1" />
                <div className="flex items-center space-x-4 text-xs">
                    <div className="flex items-center gap-2">
                        <span className={`w-2 h-2 rounded-full ${settings.enableSummarization ? 'bg-cyan-400' : 'bg-gray-600'}`} />
                        <span>TEXT</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className={`w-2 h-2 rounded-full ${settings.detectionEndpoint ? 'bg-cyan-400' : 'bg-gray-600'}`} />
                        <span>IMAGE/VIDEO</span>
                    </div>
                    <button onClick={() => setIsActive(false)} className="p-2 hover:bg-red-500/20 rounded-full transition-colors">
                        <X className="w-5 h-5" />
                    </button>
                </div>
            </div>

            {/* Reticle for both images and videos */}
            {(() => {
                const elementRect = hoveredElement?.getBoundingClientRect()
                const isOnElement = !!(hoveredElement && elementRect)

                const reticleLeft = isOnElement ? elementRect!.left : mousePos.x - 25
                const reticleTop = isOnElement ? elementRect!.top : mousePos.y - 25
                const reticleWidth = isOnElement ? elementRect!.width : 50
                const reticleHeight = isOnElement ? elementRect!.height : 50

                return (
                    <div
                        id="scanner-reticle"
                        className="fixed pointer-events-none z-[99998]"
                        style={{
                            left: reticleLeft,
                            top: reticleTop,
                            width: reticleWidth,
                            height: reticleHeight,
                            transition: isOnElement ? 'all 0.15s ease-out' : 'none',
                        }}
                    >
                        <div
                            className="absolute inset-0 border-2 transition-colors duration-200"
                            style={{
                                borderColor: isScanning ? '#facc15' : '#22d3ee',
                                boxShadow: isScanning
                                    ? '0 0 10px rgba(250, 204, 21, 0.5), inset 0 0 10px rgba(250, 204, 21, 0.1)'
                                    : '0 0 10px rgba(34, 211, 238, 0.5), inset 0 0 10px rgba(34, 211, 238, 0.1)'
                            }}
                        />
                        <div className="absolute -top-1 -left-1 w-5 h-5 border-t-2 border-l-2" style={{ borderColor: '#67e8f9' }} />
                        <div className="absolute -top-1 -right-1 w-5 h-5 border-t-2 border-r-2" style={{ borderColor: '#67e8f9' }} />
                        <div className="absolute -bottom-1 -left-1 w-5 h-5 border-b-2 border-l-2" style={{ borderColor: '#67e8f9' }} />
                        <div className="absolute -bottom-1 -right-1 w-5 h-5 border-b-2 border-r-2" style={{ borderColor: '#67e8f9' }} />

                        {!isOnElement && (
                            <>
                                <div className="absolute top-1/2 left-1/2 w-[2px] h-4 transform -translate-x-1/2 -translate-y-1/2" style={{ backgroundColor: '#22d3ee' }} />
                                <div className="absolute top-1/2 left-1/2 w-4 h-[2px] transform -translate-x-1/2 -translate-y-1/2" style={{ backgroundColor: '#22d3ee' }} />
                            </>
                        )}

                        {isScanning && isOnElement && (
                            <div className="absolute inset-0 flex items-center justify-center bg-black/20">
                                <Loader2 className="w-12 h-12 text-yellow-400 animate-spin" />
                            </div>
                        )}

                        <div className="absolute -bottom-7 left-1/2 transform -translate-x-1/2 whitespace-nowrap">
                            <div
                                className="text-[10px] uppercase tracking-widest px-2 py-1 rounded shadow-xl"
                                style={{
                                    color: '#22d3ee',
                                    backgroundColor: 'rgba(0,0,0,0.85)',
                                    border: '1px solid rgba(34, 211, 238, 0.4)'
                                }}
                            >
                                {renderSystemStatus()}
                            </div>
                        </div>

                        {/* On-video detection results */}
                        {hoveredVideo && detectionResult?.data.map((det, idx) => {
                            const x = (det.x / 100) * reticleWidth
                            const y = (det.y / 100) * reticleHeight
                            const w = (det.width / 100) * reticleWidth
                            const h = (det.height / 100) * reticleHeight

                            return (
                                <div
                                    key={idx}
                                    className="absolute"
                                    style={{
                                        left: x,
                                        top: y,
                                        width: w,
                                        height: h,
                                        border: `2px solid ${det.color}`,
                                        backgroundColor: `${det.color}26`,
                                    }}
                                >
                                    <div
                                        className="absolute -top-5 left-0 text-[10px] px-2 py-0.5 uppercase tracking-wider flex items-center gap-1 whitespace-nowrap"
                                        style={{
                                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                                            color: det.color,
                                            border: `1px solid ${det.color}66`
                                        }}
                                    >
                                        <Target className="w-2.5 h-2.5" />
                                        {det.type} {(det.confidence * 100).toFixed(0)}%
                                    </div>
                                </div>
                            )
                        })}

                        {/* On-image deep analysis sidebar (for images only) */}
                        {hoveredImage && (() => {
                            if (!detectionResult?.data) return null;

                            const analyzableDetections = detectionResult.data
                                .filter(d => !!d.analysis)
                                .sort((a, b) => a.y - b.y);

                            return (
                                <>
                                    <svg className="absolute left-full top-0 ml-0 h-full w-12 overflow-visible pointer-events-none">
                                        <AnimatePresence>
                                            {analyzableDetections.map((det, idx) => {
                                                const boxCenterY = ((det.y + det.height / 2) / 100) * reticleHeight;
                                                const boxRightEdgeX = ((det.x + det.width) / 100) * reticleWidth;

                                                const labelCenterY = idx * (90 + 24) + 45;

                                                const startX = boxRightEdgeX - reticleWidth;
                                                const startY = boxCenterY;

                                                const endX = 48;
                                                const endY = labelCenterY;

                                                return (
                                                    <motion.g
                                                        key={`line-${det.type}-${det.x}-${det.y}`}
                                                        initial={{ opacity: 0 }}
                                                        animate={{ opacity: 1 }}
                                                        exit={{ opacity: 0 }}
                                                        transition={{ duration: 0.3 }}
                                                    >
                                                        <motion.path
                                                            initial={{ pathLength: 0 }}
                                                            animate={{ pathLength: 1 }}
                                                            d={`M ${startX} ${startY} L ${endX} ${endY}`}
                                                            stroke={det.color || "#22d3ee"}
                                                            strokeWidth="1.5"
                                                            fill="none"
                                                            strokeDasharray="4 2"
                                                            className="opacity-80"
                                                        />
                                                        <motion.path
                                                            initial={{ pathLength: 0, opacity: 0 }}
                                                            animate={{
                                                                pathLength: [0, 1],
                                                                opacity: [0, 1, 0]
                                                            }}
                                                            transition={{
                                                                duration: 1.5,
                                                                repeat: Infinity,
                                                                ease: "linear"
                                                            }}
                                                            d={`M ${startX} ${startY} L ${endX} ${endY}`}
                                                            stroke={det.color || "#22d3ee"}
                                                            strokeWidth="3"
                                                            fill="none"
                                                            className="opacity-100"
                                                        />
                                                        <circle cx={startX} cy={startY} r="3" fill={det.color || "#22d3ee"} />
                                                        <circle cx={endX} cy={endY} r="3" fill={det.color || "#22d3ee"} />
                                                    </motion.g>
                                                );
                                            })}
                                        </AnimatePresence>
                                    </svg>

                                    <div className="absolute left-full top-0 ml-12 flex flex-col gap-6 w-64 pointer-events-auto">
                                        <AnimatePresence mode="popLayout">
                                            {analyzableDetections.map((det, idx) => (
                                                <motion.div
                                                    key={`label-${det.type}-${det.x}-${det.y}`}
                                                    initial={{ opacity: 0, x: 20 }}
                                                    animate={{ opacity: 1, x: 0 }}
                                                    exit={{ opacity: 0, x: 20 }}
                                                    transition={{ duration: 0.3, delay: idx * 0.05 }}
                                                    className="relative flex flex-col gap-1 p-3 transform"
                                                    style={{
                                                        backgroundColor: 'rgba(0, 0, 0, 0.85)',
                                                        color: det.color || '#22d3ee',
                                                        border: `1px solid ${det.color || '#22d3ee'}44`,
                                                        borderLeft: `4px solid ${det.color || '#22d3ee'}`,
                                                        backdropFilter: 'blur(8px)',
                                                        boxShadow: '0 4px 15px rgba(0,0,0,0.5)',
                                                        minHeight: '90px'
                                                    }}
                                                >
                                                    <div className="flex items-center justify-between border-b border-white/10 pb-1 mb-1 text-[10px] uppercase font-bold tracking-tighter">
                                                        <div className="flex items-center gap-2">
                                                            <Brain className="w-3 h-3" />
                                                            <span>{det.type} IDENTITY</span>
                                                        </div>
                                                        <span className="opacity-60">CONF: {(det.confidence * 100).toFixed(0)}%</span>
                                                    </div>

                                                    <div className="normal-case italic text-cyan-50 text-[11px] leading-relaxed">
                                                        {det.analysis === '...' ? (
                                                            <div className="flex flex-col gap-2 py-1">
                                                                <span className="flex items-center gap-2 text-[8px] uppercase tracking-tighter opacity-70">
                                                                    <span className="w-2 h-2 bg-yellow-400 rounded-full animate-ping-pong" />
                                                                    ACQUIRING DETAILS...
                                                                </span>
                                                                <div className="h-1 w-full bg-white/10 rounded-full overflow-hidden relative">
                                                                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-yellow-400 to-transparent w-2/3 animate-progress-indefinite" />
                                                                </div>
                                                            </div>
                                                        ) : (
                                                            <div className="animate-in fade-in slide-in-from-top-1 duration-500">
                                                                {det.analysis}
                                                            </div>
                                                        )}
                                                    </div>
                                                </motion.div>
                                            ))}
                                        </AnimatePresence>
                                    </div>
                                </>
                            );
                        })()}

                        {/* On-video analysis sidebar and lines */}
                        {hoveredVideo && detectionResult?.data && (() => {
                            const analyzableDetections = detectionResult.data
                                .filter(d => d.confidence > 0) // Show all detections
                                .sort((a, b) => a.y - b.y);

                            return (
                                <>
                                    <svg className="absolute left-full top-0 ml-0 h-full w-12 overflow-visible pointer-events-none">
                                        <AnimatePresence>
                                            {analyzableDetections.map((det, idx) => {
                                                const boxCenterY = ((det.y + det.height / 2) / 100) * reticleHeight;
                                                const boxRightEdgeX = ((det.x + det.width) / 100) * reticleWidth;

                                                const labelCenterY = idx * (90 + 24) + 45;

                                                const startX = boxRightEdgeX - reticleWidth;
                                                const startY = boxCenterY;

                                                const endX = 48;
                                                const endY = labelCenterY;

                                                return (
                                                    <motion.g
                                                        key={`line-${det.type}-${det.x}-${det.y}`}
                                                        initial={{ opacity: 0 }}
                                                        animate={{ opacity: 1 }}
                                                        exit={{ opacity: 0 }}
                                                        transition={{ duration: 0.3 }}
                                                    >
                                                        <motion.path
                                                            initial={{ pathLength: 0 }}
                                                            animate={{ pathLength: 1 }}
                                                            d={`M ${startX} ${startY} L ${endX} ${endY}`}
                                                            stroke={det.color || "#22d3ee"}
                                                            strokeWidth="1.5"
                                                            fill="none"
                                                            strokeDasharray="4 2"
                                                            className="opacity-80"
                                                        />
                                                        <motion.path
                                                            initial={{ pathLength: 0, opacity: 0 }}
                                                            animate={{
                                                                pathLength: [0, 1],
                                                                opacity: [0, 1, 0]
                                                            }}
                                                            transition={{
                                                                duration: 1.5,
                                                                repeat: Infinity,
                                                                ease: "linear"
                                                            }}
                                                            d={`M ${startX} ${startY} L ${endX} ${endY}`}
                                                            stroke={det.color || "#22d3ee"}
                                                            strokeWidth="3"
                                                            fill="none"
                                                            className="opacity-100"
                                                        />
                                                        <circle cx={startX} cy={startY} r="3" fill={det.color || "#22d3ee"} />
                                                        <circle cx={endX} cy={endY} r="3" fill={det.color || "#22d3ee"} />
                                                    </motion.g>
                                                );
                                            })}
                                        </AnimatePresence>
                                    </svg>

                                    <div className="absolute left-full top-0 ml-12 flex flex-col gap-6 w-64 pointer-events-auto">
                                        <AnimatePresence mode="popLayout">
                                            {analyzableDetections.map((det, idx) => (
                                                <motion.div
                                                    key={`label-${det.type}-${det.x}-${det.y}`}
                                                    initial={{ opacity: 0, x: 20 }}
                                                    animate={{ opacity: 1, x: 0 }}
                                                    exit={{ opacity: 0, x: 20 }}
                                                    transition={{ duration: 0.3, delay: idx * 0.05 }}
                                                    className="relative flex flex-col gap-1 p-3 transform"
                                                    style={{
                                                        backgroundColor: 'rgba(0, 0, 0, 0.85)',
                                                        color: det.color || '#22d3ee',
                                                        border: `1px solid ${det.color || '#22d3ee'}44`,
                                                        borderLeft: `4px solid ${det.color || '#22d3ee'}`,
                                                        backdropFilter: 'blur(8px)',
                                                        boxShadow: '0 4px 15px rgba(0,0,0,0.5)',
                                                        minHeight: '90px'
                                                    }}
                                                >
                                                    <div className="flex items-center justify-between border-b border-white/10 pb-1 mb-1 text-[10px] uppercase font-bold tracking-tighter">
                                                        <div className="flex items-center gap-2">
                                                            <Target className="w-3 h-3" />
                                                            <span>{det.type}</span>
                                                        </div>
                                                        <span className="opacity-60">CONF: {(det.confidence * 100).toFixed(0)}%</span>
                                                    </div>

                                                    <div className="normal-case italic text-cyan-50 text-[11px] leading-relaxed">
                                                        {det.analysis === '...' ? (
                                                            <div className="flex flex-col gap-2 py-1">
                                                                <span className="flex items-center gap-2 text-[8px] uppercase tracking-tighter opacity-70">
                                                                    <span className="w-2 h-2 bg-yellow-400 rounded-full animate-ping-pong" />
                                                                    ANALYZING...
                                                                </span>
                                                                <div className="h-1 w-full bg-white/10 rounded-full overflow-hidden relative">
                                                                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-yellow-400 to-transparent w-2/3 animate-progress-indefinite" />
                                                                </div>
                                                            </div>
                                                        ) : (
                                                            <div className="animate-in fade-in slide-in-from-top-1 duration-500">
                                                                {det.analysis || 'No detailed analysis.'}
                                                            </div>
                                                        )}
                                                    </div>
                                                </motion.div>
                                            ))}
                                        </AnimatePresence>
                                    </div>
                                </>
                            );
                        })()}
                    </div>
                )
            })()}

            {/* Text Summary Overlay */}
            {settings.enableSummarization && (
                <SummaryOverlay
                    selection={selection}
                    summaryResult={summaryResult}
                    isSummarizing={isSummarizing}
                    error={summaryError}
                    mousePos={activeMousePos}
                />
            )}

            <div className="absolute bottom-0 left-0 right-0 h-12 bg-black/80 flex items-center justify-between px-8 text-xs border-t border-cyan-900/50">
                <div className="font-mono">
                    COORDS: {activeMousePos.x.toFixed(0)} : {activeMousePos.y.toFixed(0)} | REF: {systemStatus}
                </div>
                <div className="flex items-center space-x-2">
                    <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                    <span>SYSTEM NORMAL</span>
                </div>
            </div>
        </div>
    )
}

export default ScannerHUD
