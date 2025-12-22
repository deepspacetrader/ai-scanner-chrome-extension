/**
 * Audio Service using ZzFX npm package
 * https://github.com/KilledByAPixel/ZzFX
 */

import { zzfx } from 'zzfx';

// Preset Sounds - ZzFX parameters
// [volume, randomness, frequency, attack, sustain, release, shape, shapeCurve, slide, deltaSlide, pitchJump, pitchJumpTime, repeatTime, noise, modulation, bitCrush, delay, sustainVolume, decay, tremolo]
export const SOUNDS = {
    SCAN: [1.0, 0.1, 80, 0.1, 0.5, 0.2, 1, 0.5, 0, 0, 0, 0, 0, 0, 0.2, 0, 0.1, 0.5, 0.1],
    LOCK: [1.0, 0, 800, 0.01, 0.05, 0.1, 0, 1.5, 0, 0, 10, 0.01, 0, 0, 0, 0, 0, 0.5, 0],
    GLITCH: [1.0, 0.5, 50, 0, 0.01, 0.05, 4, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0.8, 0],
    UI_CLICK: [1.0, 0, 400, 0, 0.05, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0],
};

export const playSound = (params: number[], masterVolume: number = 1) => {
    try {
        if (masterVolume <= 0) return;

        // Apply master volume as a multiplier to the sound's inherent volume
        const modifiedParams = [...params];
        modifiedParams[0] = (modifiedParams[0] || 1.0) * masterVolume;

        // Use the official zzfx function from the npm package
        zzfx(...modifiedParams);
    } catch (e) {
        console.warn('Audio play failed', e);
    }
};

export const audioService = {
    playScan: (volume: number = 1) => playSound(SOUNDS.SCAN, volume),
    playLock: (volume: number = 1) => playSound(SOUNDS.LOCK, volume),
    playGlitch: (volume: number = 1) => playSound(SOUNDS.GLITCH, volume),
    playClick: (volume: number = 1) => playSound(SOUNDS.UI_CLICK, volume),
};
