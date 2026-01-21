export interface Settings {
    triggerInput: string;
    detectionEndpoint: string;
    detectionModel: string; // New field for model selection
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
    visionModel: string;
}

export const DEFAULT_SETTINGS: Settings = {
    triggerInput: "keyboard:Alt",
    detectionEndpoint: "http://localhost:8001/api/detect-base64",
    detectionModel: "segmentation", // Default to segmentation model
    enableSummarization: true,
    summarizationEndpoint: "http://localhost:8001/api/summarize",
    summarizationModel: "zai-org/glm-4.6v-flash",
    minSummaryChars: 40,
    toggleActivation: true,
    saveScannedImages: false,
    enableDeepAnalysis: false,
    enableEnhancedDescription: true,
    deepAnalysisThreshold: 0.85,
    categoryThresholds: {
        "Humans": 0.70,
        "Vehicles": 0.90,
        "Animals": 0.90,
        "Outdoors": 0.90,
        "Accessories": 0.90,
        "Sports": 0.90,
        "Household": 0.90,
        "Food": 0.90,
        "Electronics": 0.90,
        "Misc": 0.90
    },
    enableSound: true,
    soundVolume: 1,
    visionModel: 'florence2'
};
