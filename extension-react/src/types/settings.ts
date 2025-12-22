export interface Settings {
    triggerInput: string;
    detectionEndpoint: string;
    showCrawlingLines: boolean;
    enableSummarization: boolean;
    summarizationEndpoint: string;
    summarizationModel: string;
    minSummaryChars: number;
    toggleActivation: boolean;
    saveScannedImages: boolean;
    enableDeepAnalysis: boolean;
    enableEnhancedDescription: boolean;
    deepAnalysisThreshold: number;
    categoryThresholds: Record<string, number>;
    enableSound: boolean;
    soundVolume: number;
}

export const DEFAULT_SETTINGS: Settings = {
    triggerInput: "keyboard:Alt",
    detectionEndpoint: "http://localhost:8001/api/detect-base64",
    showCrawlingLines: true,
    enableSummarization: true,
    summarizationEndpoint: "http://localhost:8001/api/summarize",
    summarizationModel: "Qwen/Qwen2.5-0.5B-Instruct",
    minSummaryChars: 40,
    toggleActivation: true,
    saveScannedImages: false,
    enableDeepAnalysis: false,
    enableEnhancedDescription: true,
    deepAnalysisThreshold: 0.85,
    categoryThresholds: {
        "Humans": 0.85,
        "Vehicles": 0.85,
        "Animals": 0.85,
        "Outdoors": 0.85,
        "Accessories": 0.85,
        "Sports": 0.85,
        "Household": 0.85,
        "Food": 0.85,
        "Electronics": 0.85,
        "Misc": 0.85
    },
    enableSound: true,
    soundVolume: 0.5
};
