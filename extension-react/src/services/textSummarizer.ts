// Text Summarization Service
// Handles summarization requests to the local Python backend

export interface SummaryResult {
    summary: string
    model: string
}

export interface SummarySettings {
    endpoint: string
    model: string
    minChars: number
    verySimpleSummary?: boolean
    systemPrompt?: string
    stream?: boolean
}

export interface StreamingSummaryResult {
    text: string
    finished: boolean
    model: string
}

class TextSummarizer {
    private cache: Map<string, SummaryResult> = new Map()
    private readonly maxCacheSize = 30
    private readonly maxInputChars = 5000
    private abortController: AbortController | null = null

    /**
     * Generate a cache key from text
     */
    generateKey(text: string): string {
        const normalized = text.trim().replace(/\s+/g, ' ')
        let hash = 0
        for (let i = 0; i < normalized.length; i++) {
            hash = ((hash << 5) - hash) + normalized.charCodeAt(i)
            hash |= 0 // Convert to 32bit integer
        }
        return `${normalized.length}:${Math.abs(hash)}`
    }

    /**
     * Post-process summary to remove thinking tags or extra noise
     */
    private postProcess(text: string): string {
        if (!text) return ''

        let cleaned = text

        // Remove <thinking>...</thinking> blocks
        cleaned = cleaned.replace(/<thinking>[\s\S]*?<\/thinking>/gi, '')

        // Remove any common LLM prefixes
        cleaned = cleaned.replace(/^(Summary|Here is a summary):/i, '')

        return cleaned.trim()
    }

    /**
     * Summarize text using local Python backend with streaming support
     * @param wakeRetried internal: true after one wake-and-retry to avoid infinite loop
     */
    async summarize(text: string, settings: SummarySettings): Promise<SummaryResult> {
        // Truncate if too long
        const truncated = text.slice(0, this.maxInputChars)
        const key = this.generateKey(truncated)

        // Check cache
        const cached = this.cache.get(key)
        if (cached) {
            return cached
        }

        // Cancel any pending request
        if (this.abortController) {
            this.abortController.abort()
        }
        this.abortController = new AbortController()
        const signal = this.abortController.signal

        try {
            // Make API request to our local server
            // The endpoint is typically http://localhost:8001/api/summarize
            const response = await fetch(settings.endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    ...settings,
                    text: truncated,
                    endpoint: undefined, // Don't send endpoint to itself
                    stream: settings.stream || false
                }),
                signal,
            })

            if (!response.ok) {
                // Ignore errors from aborted requests
                if (signal.aborted) return new Promise(() => { })
                throw new Error(`Summarization request failed (${response.status})`)
            }

            if (settings.stream) {
                // Handle streaming response
                const reader = response.body?.getReader()
                if (!reader) {
                    throw new Error('No response body for streaming')
                }

                const decoder = new TextDecoder()
                let fullText = ''

                while (true) {
                    const { done, value } = await reader.read()
                    if (done) break

                    const chunk = decoder.decode(value, { stream: true })
                    const lines = chunk.split('\n')

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6))
                                if (data.cancelled) {
                                    return new Promise(() => { }) // Stay in pending state as we are cancelled
                                }
                                if (data.error) {
                                    throw new Error(data.error)
                                }
                                if (data.text) {
                                    fullText = data.text
                                }
                            } catch (e) {
                                // Ignore JSON parse errors for partial chunks
                            }
                        }
                    }
                }

                const summaryText = this.postProcess(fullText)

                if (!summaryText) {
                    throw new Error('Local summarizer returned an empty response')
                }

                const result: SummaryResult = {
                    summary: summaryText,
                    model: settings.model || 'Qwen2.5-0.5B-Instruct',
                }

                // Cache result
                this.cache.set(key, result)
                if (this.cache.size > this.maxCacheSize) {
                    const firstKey = this.cache.keys().next().value
                    if (firstKey) {
                        this.cache.delete(firstKey)
                    }
                }

                return result
            } else {
                // Handle non-streaming response (original behavior)
                const data = await response.json()

                // If the backend returned a cancellation notice or we are aborted locally
                if (data.cancelled || signal.aborted) {
                    return new Promise(() => { }) // Stay in pending state as we are cancelled
                }

                const summaryText = this.postProcess(data.summary || '')

                if (!summaryText) {
                    throw new Error('Local summarizer returned an empty response')
                }

                const result: SummaryResult = {
                    summary: summaryText,
                    model: settings.model || 'Qwen2.5-0.5B-Instruct',
                }

                // Cache result
                this.cache.set(key, result)
                if (this.cache.size > this.maxCacheSize) {
                    const firstKey = this.cache.keys().next().value
                    if (firstKey) {
                        this.cache.delete(firstKey)
                    }
                }

                return result
            }
        } catch (error: any) {
            if (error.name === 'AbortError') {
                return new Promise(() => { })
            }
            throw error
        } finally {
            if (this.abortController?.signal === signal) {
                this.abortController = null
            }
        }
    }

    /**
     * Stream summarize text with callback for real-time updates
     * @param wakeRetried internal: true after one wake-and-retry
     */
    async streamSummarize(
        text: string, 
        settings: SummarySettings, 
        onChunk: (chunk: StreamingSummaryResult) => void
    ): Promise<SummaryResult> {
        // Truncate if too long
        const truncated = text.slice(0, this.maxInputChars)
        const key = this.generateKey(truncated)

        // Cancel any pending request
        if (this.abortController) {
            this.abortController.abort()
        }
        this.abortController = new AbortController()
        const signal = this.abortController.signal

        try {
            // Make API request to our local server with streaming
            const response = await fetch(settings.endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    ...settings,
                    text: truncated,
                    endpoint: undefined, // Don't send endpoint to itself
                    stream: true
                }),
                signal,
            })

            if (!response.ok) {
                if (signal.aborted) return new Promise(() => { })
                throw new Error(`Summarization request failed (${response.status})`)
            }

            const reader = response.body?.getReader()
            if (!reader) {
                throw new Error('No response body for streaming')
            }

            const decoder = new TextDecoder()
            let finalText = ''

            while (true) {
                const { done, value } = await reader.read()
                if (done) break

                const chunk = decoder.decode(value, { stream: true })
                const lines = chunk.split('\n')

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6))
                            if (data.cancelled) {
                                return new Promise(() => { })
                            }
                            if (data.error) {
                                throw new Error(data.error)
                            }
                            if (data.text !== undefined) {
                                const chunkResult: StreamingSummaryResult = {
                                    text: data.text,
                                    finished: data.finished || false,
                                    model: data.model || 'Qwen2.5'
                                }
                                onChunk(chunkResult)
                                if (data.finished) {
                                    finalText = data.text
                                }
                            }
                        } catch (e) {
                            // Ignore JSON parse errors for partial chunks
                        }
                    }
                }
            }

            const summaryText = this.postProcess(finalText)

            if (!summaryText) {
                throw new Error('Local summarizer returned an empty response')
            }

            const result: SummaryResult = {
                summary: summaryText,
                model: settings.model || 'Qwen2.5-0.5B-Instruct',
            }

            // Cache result
            this.cache.set(key, result)
            if (this.cache.size > this.maxCacheSize) {
                const firstKey = this.cache.keys().next().value
                if (firstKey) {
                    this.cache.delete(firstKey)
                }
            }

            return result
        } catch (error: any) {
            if (error.name === 'AbortError') {
                return new Promise(() => { })
            }
            throw error
        } finally {
            if (this.abortController?.signal === signal) {
                this.abortController = null
            }
        }
    }

    /**
     * Clear cache
     */
    clearCache() {
        this.cache.clear()
    }
}

export const textSummarizer = new TextSummarizer()
