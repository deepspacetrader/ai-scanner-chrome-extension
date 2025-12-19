import { useState, useEffect } from 'react'
import './App.css'
import { ScanEye } from 'lucide-react'

// Define types for settings logic
interface Settings {
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
}

const DEFAULT_SETTINGS: Settings = {
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
  }
};

function App() {
  const [status, setStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking')
  const [message, setMessage] = useState<string>('');
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [activeTabId, setActiveTabId] = useState<number | null>(null);

  useEffect(() => {
    // Load settings
    if (chrome?.storage?.sync) {
      chrome.storage.sync.get(
        Object.keys(DEFAULT_SETTINGS),
        (result) => {
          const mergedSettings = { ...DEFAULT_SETTINGS, ...result };
          if (typeof mergedSettings.minSummaryChars !== 'number') mergedSettings.minSummaryChars = DEFAULT_SETTINGS.minSummaryChars;

          // Auto-migrate from Granite to Qwen if old model is detected
          if (mergedSettings.summarizationModel.toLowerCase().includes('granite')) {
            mergedSettings.summarizationModel = DEFAULT_SETTINGS.summarizationModel;
          }

          setSettings(mergedSettings as Settings);
          testConnection(mergedSettings.detectionEndpoint);
        }
      );
    } else {
      setStatus('disconnected');
      setMessage('Dev Mode: Storage API unavailable');
    }

    if (chrome?.tabs) {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs[0]?.id) setActiveTabId(tabs[0].id);
      });
    }
  }, []);

  const handleChange = (key: keyof Settings, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  const saveSettings = () => {
    if (chrome?.storage?.sync) {
      chrome.storage.sync.set(settings, () => {
        setMessage('Settings saved successfully!');

        const hasEndpoint = Boolean(settings.detectionEndpoint && settings.detectionEndpoint.trim());
        let iconState = 'images';
        if (status !== 'connected') iconState = 'offline';
        else if (settings.enableSummarization && hasEndpoint) iconState = 'both';
        else if (settings.enableSummarization && !hasEndpoint) iconState = 'text';

        chrome.runtime.sendMessage({ type: "UPDATE_ICON_STATE", state: iconState });

        setTimeout(() => setMessage(''), 3000);
      });
    } else {
      setMessage('Settings saved (mock)!');
    }
  };

  const testConnection = async (endpoint: string = settings.detectionEndpoint) => {
    setStatus('checking');
    try {
      const statusUrl = endpoint.replace("/api/detect-base64", "/api/status");
      const res = await fetch(statusUrl);
      if (res.ok) {
        setStatus('connected');
      } else {
        setStatus('disconnected');
      }
    } catch (e) {
      setStatus('disconnected');
    }
  };

  const handleScanPage = () => {
    if (activeTabId && chrome?.tabs) {
      chrome.tabs.sendMessage(activeTabId, { type: "SCAN_FULL_PAGE_TEXT" });
      window.close();
    }
  };

  const triggerOptions = [
    { label: 'Shift', value: 'keyboard:Shift' },
    { label: 'Ctrl', value: 'keyboard:Control' },
    { label: 'Alt', value: 'keyboard:Alt' },
    { label: 'Left Click', value: 'mouse:0' },
    { label: 'Middle Click', value: 'mouse:1' },
    { label: 'Right Click', value: 'mouse:2' },
  ];

  return (
    <div className="theme-container">
      <div className="theme-header">
        <h1 className="theme-title">AI Scanner Tool</h1>
        <p className="theme-subtitle">AI enhanced web browsing with configurable activation</p>
      </div>

      <div className="theme-content">

        {/* Full Page Scan button (if enabled) */}
        {settings.enableSummarization && (
          <div className="theme-setting">
            <button
              onClick={handleScanPage}
              className="theme-btn theme-btn-primary"
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
            >
              <ScanEye size={14} />
              Scan Full Page Text
            </button>
          </div>
        )}

        <div className="theme-setting">
          <label className="theme-label">Detection API URL:</label>
          <input
            type="text"
            className="theme-input"
            value={settings.detectionEndpoint}
            onChange={(e) => handleChange('detectionEndpoint', e.target.value)}
            placeholder="http://localhost:8001/api/detect-base64"
          />
        </div>

        <div className="theme-setting">
          <label className="theme-label">Activation Method:</label>
          <div className="theme-grid">
            {triggerOptions.map((opt) => (
              <div
                key={opt.value}
                className={`theme-option ${settings.triggerInput === opt.value ? 'selected' : ''}`}
                onClick={() => handleChange('triggerInput', opt.value)}
              >
                {opt.label}
              </div>
            ))}
          </div>
        </div>

        <div className="theme-setting">
          <label className="theme-label">Activation Mode:</label>
          <div className="theme-grid" style={{ gridTemplateColumns: 'repeat(2, 1fr)' }}>
            <div
              className={`theme-option ${!settings.toggleActivation ? 'selected' : ''}`}
              onClick={() => handleChange('toggleActivation', false)}
            >
              Hold
            </div>
            <div
              className={`theme-option ${settings.toggleActivation ? 'selected' : ''}`}
              onClick={() => handleChange('toggleActivation', true)}
            >
              Toggle
            </div>
          </div>
        </div>

        <div className="theme-setting">
          <label className="theme-label" style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={settings.showCrawlingLines}
              onChange={(e) => handleChange('showCrawlingLines', e.target.checked)}
            />
            Show crawling lines
          </label>
        </div>

        <div className="theme-setting">
          <label className="theme-label" style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={settings.saveScannedImages}
              onChange={(e) => handleChange('saveScannedImages', e.target.checked)}
            />
            Save scanned images to server
          </label>
        </div>

        <div className="theme-setting">
          <label className="theme-label" style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={settings.enableEnhancedDescription}
              onChange={(e) => handleChange('enableEnhancedDescription', e.target.checked)}
            />
            Enhanced Image Description (LLM Refinement)
          </label>
        </div>

        <div className="theme-setting">
          <label className="theme-label" style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={settings.enableDeepAnalysis}
              onChange={(e) => handleChange('enableDeepAnalysis', e.target.checked)}
            />
            Enable Deep Analysis (Florence-2)
          </label>
        </div>

        {settings.enableDeepAnalysis && (
          <div className="theme-content" style={{ paddingLeft: '10px', borderLeft: '2px solid #ddd', marginBottom: '15px' }}>
            <div className="theme-setting">
              <label className="theme-label">VLM Analysis Threshold (Global):</label>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={settings.deepAnalysisThreshold}
                  onChange={(e) => handleChange('deepAnalysisThreshold', parseFloat(e.target.value))}
                  style={{ flex: 1 }}
                />
                <span style={{ fontSize: '12px', width: '40px' }}>{(settings.deepAnalysisThreshold * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>
        )}

        {/* Category Thresholds List */}
        <div className="theme-setting">
          <label className="theme-label" style={{ marginBottom: '4px', display: 'block' }}>VLM Analysis Thresholds (per category):</label>
          <p style={{ fontSize: '10px', color: '#666', marginBottom: '10px', lineHeight: '1.2' }}>
            YOLO detections are always visible. Adjust these to control when Florence-2 analysis is triggered for entire categories.
          </p>
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '8px',
            backgroundColor: 'rgba(0,0,0,0.05)',
            padding: '10px',
            borderRadius: '4px',
            border: '1px solid #ddd'
          }}>
            {Object.entries(settings.categoryThresholds).map(([cat, val]) => (
              <div key={cat} style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                <span style={{ fontSize: '10px', textTransform: 'uppercase', fontWeight: 'bold' }}>{cat}</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={val}
                    onChange={(e) => {
                      const newThresholds = { ...settings.categoryThresholds, [cat]: parseFloat(e.target.value) };
                      handleChange('categoryThresholds', newThresholds);
                    }}
                    style={{ flex: 1 }}
                  />
                  <span style={{ fontSize: '10px', width: '25px' }}>{(val * 100).toFixed(0)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Summarization Section - Collapsible or Inline */}
        <div className="theme-setting">
          <label className="theme-label" style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={settings.enableSummarization}
              onChange={(e) => handleChange('enableSummarization', e.target.checked)}
            />
            Enable Text Summarization
          </label>
        </div>

        {settings.enableSummarization && (
          <div className="theme-content" style={{ paddingLeft: '10px', borderLeft: '2px solid #ddd' }}>
            <div className="theme-setting">
              <label className="theme-label">Local Summarization API URL:</label>
              <input
                type="text"
                className="theme-input"
                value={settings.summarizationEndpoint}
                onChange={(e) => handleChange('summarizationEndpoint', e.target.value)}
              />
            </div>

            <div className="theme-setting">
              <label className="theme-label">Model ID:</label>
              <input
                type="text"
                className="theme-input"
                value={settings.summarizationModel}
                onChange={(e) => handleChange('summarizationModel', e.target.value)}
              />
            </div>
          </div>
        )}

        <div className="theme-setting">
          <label className="theme-label">Connection Status:</label>
          <div className={`theme-status ${status}`}>
            {status === 'connected' ? 'Connected to YOLO server' :
              status === 'checking' ? 'Checking...' :
                'Disconnected - check API URL'}
          </div>
        </div>

        <div className="theme-buttons">
          <button className="theme-btn theme-btn-primary" onClick={saveSettings}>Save</button>
          <button className="theme-btn theme-btn-secondary" onClick={() => testConnection()}>Test</button>
        </div>

        {
          message && (
            <div style={{ textAlign: 'center', fontSize: '12px', color: '#28a745', marginTop: '5px' }}>
              {message}
            </div>
          )
        }

        <div className="theme-setting">
          <p style={{ fontSize: '11px', color: '#666', margin: '0', lineHeight: '1.4' }}>
            Hold the selected input and hover over images to see AI detection results in a floating popup.
            Make sure the YOLO server is running on the specified API URL.
          </p>
        </div>

      </div >
    </div >
  )
}

export default App
