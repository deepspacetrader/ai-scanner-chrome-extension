/** @type {import('tailwindcss').Config} */
export default {
    important: '#ai-scanner-tool-root',
    // Disable preflight (base reset styles) to avoid affecting host pages
    corePlugins: {
        preflight: false,
    },
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                cyber: {
                    bg: '#050510',
                    panel: 'rgba(10, 10, 30, 0.8)',
                    primary: '#00f0ff', // Cyan
                    secondary: '#7000ff', // Purple
                    accent: '#ff003c', // Red
                    dim: 'rgba(0, 240, 255, 0.1)',
                }
            },
            animation: {
                'scanline': 'scanline 2s linear infinite',
                'pulse-fast': 'pulse 1s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                'progress-indefinite': 'progress-indefinite 1.5s ease-in-out infinite',
                'ping-pong': 'ping-pong 1s ease-in-out infinite',
            },
            keyframes: {
                scanline: {
                    '0%': { transform: 'translateY(-100%)' },
                    '100%': { transform: 'translateY(100vh)' },
                },
                'progress-indefinite': {
                    '0%': { transform: 'translateX(-100%)' },
                    '100%': { transform: 'translateX(100%)' },
                },
                'ping-pong': {
                    '0%, 100%': { transform: 'translateX(0)' },
                    '50%': { transform: 'translateX(10px)' },
                }
            }
        },
    },
    plugins: [],
}
