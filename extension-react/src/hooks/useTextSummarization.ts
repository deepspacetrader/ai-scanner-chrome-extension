import { useState, useEffect, useCallback, useRef } from 'react'
import { textSummarizer, type SummaryResult, type SummarySettings, type StreamingSummaryResult } from '../services/textSummarizer'

export interface TextSelection {
    text: string
    rawText: string
    preview: string
    rect: DOMRect | null
}

export function useTextSummarization(isActive: boolean, settings: SummarySettings) {
    const [selection, setSelection] = useState<TextSelection | null>(null)
    const [summaryResult, setSummaryResult] = useState<SummaryResult | null>(null)
    const [streamingText, setStreamingText] = useState<string>('')
    const [isSummarizing, setIsSummarizing] = useState(false)
    const [error, setError] = useState<string>('')
    const currentKeyRef = useRef<string>('')

    /**
     * Build selection preview (truncated text for display)
     */
    const buildPreview = useCallback((text: string): string => {
        if (!text) return ''
        const trimmed = text.trim().replace(/\s+/g, ' ')
        const maxPreview = 160
        return trimmed.length > maxPreview
            ? `${trimmed.slice(0, maxPreview)}…`
            : trimmed
    }, [])

    /**
     * Compute selection bounding rect
     */
    const computeSelectionRect = useCallback((): DOMRect | null => {
        try {
            const sel = document.getSelection()
            if (!sel || sel.rangeCount === 0) return null

            const range = sel.getRangeAt(0)
            const rect = range.getBoundingClientRect()
            if (rect && (rect.width || rect.height)) {
                return rect
            }

            const clientRects = range.getClientRects()
            if (clientRects && clientRects.length) {
                return clientRects[0] as DOMRect
            }
        } catch (error) {
            // Ignore selection measurement errors
        }
        return null
    }, [])

    /**
     * Handle mouse up - check for text selection and trigger summarization
     */
    const handleMouseUp = useCallback(() => {
        if (!isActive || !settings) return

        const sel = document.getSelection()
        if (!sel || sel.isCollapsed) {
            // No selection or collapsed - clear state
            setSelection(null)
            setSummaryResult(null)
            setError('')
            currentKeyRef.current = ''
            return
        }

        const text = sel.toString().trim()

        // Check minimum character requirement
        if (!text || text.length < settings.minChars) {
            setSelection(null)
            setSummaryResult(null)
            setError('')
            currentKeyRef.current = ''
            return
        }

        const truncated = text.slice(0, 5000)
        const selectionData: TextSelection = {
            text: truncated,
            rawText: text,
            preview: buildPreview(text),
            rect: computeSelectionRect(),
        }

        setSelection(selectionData)

        // Start summarization
        const key = textSummarizer.generateKey(truncated)
        currentKeyRef.current = key

        setIsSummarizing(true)
        setError('')
        setStreamingText('')

        // Use streaming if enabled
        const useStreaming = settings.stream
        console.log('useTextSummarization: Starting summarization, stream:', settings.stream, 'model_provider:', settings.model_provider, 'useStreaming:', useStreaming)
        if (useStreaming) {
            let finished = false
            textSummarizer.streamSummarize(truncated, settings, (chunk: StreamingSummaryResult) => {
                if (currentKeyRef.current === key && !finished) {
                    setStreamingText(chunk.text)
                    if (chunk.finished) {
                        finished = true
                        setStreamingText('')
                        setSummaryResult({
                            summary: chunk.text,
                            model: chunk.model
                        })
                    }
                }
            })
            .then((result) => {
                if (currentKeyRef.current === key) {
                    setSummaryResult(result)
                    setStreamingText('')
                    setError('')
                }
            })
            .catch((err) => {
                if (currentKeyRef.current === key) {
                    setError(err?.message || 'Summarization failed')
                    setSummaryResult(null)
                    setStreamingText('')
                }
            })
            .finally(() => {
                if (currentKeyRef.current === key) {
                    setIsSummarizing(false)
                }
            })
        } else {
            console.log('useTextSummarization: Calling non-streaming summarize')
            textSummarizer.summarize(truncated, settings)
                .then((result) => {
                    console.log('useTextSummarization: Summarize result:', result)
                    if (currentKeyRef.current === key) {
                        setSummaryResult(result)
                        setError('')
                        setStreamingText('')
                    }
                })
                .catch((err) => {
                    if (currentKeyRef.current === key) {
                        setError(err?.message || 'Summarization failed')
                        setSummaryResult(null)
                        setStreamingText('')
                    }
                })
                .finally(() => {
                    if (currentKeyRef.current === key) {
                        setIsSummarizing(false)
                    }
                })
        }
    }, [isActive, settings, buildPreview, computeSelectionRect])

    /**
     * Extract text from page
     */
    const extractPageText = useCallback((): string => {
        const clone = document.body.cloneNode(true) as HTMLElement
        const scripts = clone.querySelectorAll('script, style, noscript')
        scripts.forEach((el) => el.remove())

        const text = clone.innerText || clone.textContent || ''
        return text
            .replace(/\s+/g, ' ')
            .replace(/\n\s*\n/g, '\n')
            .trim()
    }, [])

    /**
     * Scan full page text
     */
    const scanFullPage = useCallback(async () => {
        if (!isActive || !settings) return

        try {
            const fullText = extractPageText()
            if (!fullText || fullText.length < settings.minChars) {
                setError('Page text is too short or empty')
                return
            }

            const truncated = fullText.slice(0, 5000)
            const key = textSummarizer.generateKey(truncated)
            currentKeyRef.current = key

            const selectionData: TextSelection = {
                text: truncated,
                rawText: fullText,
                preview: buildPreview(fullText),
                rect: null,
            }

            setSelection(selectionData)
            setIsSummarizing(true)
            setError('')
            setStreamingText('')

            // Use streaming if enabled, otherwise fall back to regular summarization
            if (settings.stream) {
                textSummarizer.streamSummarize(truncated, settings, (chunk: StreamingSummaryResult) => {
                    if (currentKeyRef.current === key) {
                        setStreamingText(chunk.text)
                        if (chunk.finished) {
                            setStreamingText('')
                            setSummaryResult({
                                summary: chunk.text,
                                model: chunk.model
                            })
                        }
                    }
                })
                .then((result) => {
                    if (currentKeyRef.current === key) {
                        setSummaryResult(result)
                        setStreamingText('')
                        setError('')
                    }
                })
                .catch((err) => {
                    if (currentKeyRef.current) {
                        setError(err?.message || 'Failed to scan page text')
                        setSummaryResult(null)
                        setStreamingText('')
                    }
                })
                .finally(() => {
                    if (currentKeyRef.current) {
                        setIsSummarizing(false)
                    }
                })
            } else {
                const result = await textSummarizer.summarize(truncated, settings)
                if (currentKeyRef.current === key) {
                    setSummaryResult(result)
                    setError('')
                    setStreamingText('')
                }
            }
        } catch (err: any) {
            if (currentKeyRef.current) {
                setError(err?.message || 'Failed to scan page text')
                setSummaryResult(null)
            }
        } finally {
            if (currentKeyRef.current) {
                setIsSummarizing(false)
            }
        }
    }, [isActive, settings, extractPageText, buildPreview])

    /**
     * Clear summary state
     */
    const clearSummary = useCallback(() => {
        setSelection(null)
        setSummaryResult(null)
        setStreamingText('')
        setError('')
        setIsSummarizing(false)
        currentKeyRef.current = ''
    }, [])

    // Listen for mouse up to detect end of text selection
    useEffect(() => {
        if (!isActive) {
            clearSummary()
            return
        }

        document.addEventListener('mouseup', handleMouseUp)
        return () => {
            document.removeEventListener('mouseup', handleMouseUp)
        }
    }, [isActive, handleMouseUp, clearSummary])

    // Listen for messages from popup (scan full page)
    useEffect(() => {
        if (!isActive) return

        const handleMessage = (request: any) => {
            if (request.type === 'SCAN_FULL_PAGE_TEXT') {
                scanFullPage()
            }
        }

        chrome.runtime?.onMessage.addListener(handleMessage)
        return () => {
            chrome.runtime?.onMessage.removeListener(handleMessage)
        }
    }, [isActive, scanFullPage])

    return {
        selection,
        summaryResult,
        streamingText,
        isSummarizing,
        error,
        scanFullPage,
        clearSummary,
    }
}
