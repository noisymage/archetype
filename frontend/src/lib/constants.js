/**
 * Reference image slot configurations
 * Shared between CreateProjectWizard and EditReferencesModal
 */

export const HEAD_SLOTS = [
    // Top Row (Diagonal Up)
    { key: 'head_up_l', label: 'Look Up Left', icon: '↖', description: 'Pitch +30°, Yaw -45°', required: false, optional: true },
    { key: 'head_up', label: 'Look Up', icon: '⮝', description: 'Pitch +30°', required: false, optional: true },
    { key: 'head_up_r', label: 'Look Up Right', icon: '↗', description: 'Pitch +30°, Yaw +45°', required: false, optional: true },

    // Middle Row (Horizontal)
    { key: 'head_90l', label: '90° Left', icon: '⮜', description: 'Left Profile', required: false, optional: true },
    { key: 'head_45l', label: '45° Left', icon: '←', description: "Viewer's left", required: true },
    { key: 'head_front', label: 'Front', icon: '⚫', description: 'Face camera', required: true },
    { key: 'head_45r', label: '45° Right', icon: '→', description: "Viewer's right", required: true },
    { key: 'head_90r', label: '90° Right', icon: '⮞', description: 'Right Profile', required: false, optional: true },

    // Bottom Row (Diagonal Down)
    { key: 'head_down_l', label: 'Look Down Left', icon: '↙', description: 'Pitch -30°, Yaw -45°', required: false, optional: true },
    { key: 'head_down', label: 'Look Down', icon: '⮟', description: 'Pitch -30°', required: false, optional: true },
    { key: 'head_down_r', label: 'Look Down Right', icon: '↘', description: 'Pitch -30°, Yaw +45°', required: false, optional: true },
];

export const BODY_SLOTS = [
    { key: 'body_front', label: 'A-Pose Front', icon: '╋', required: true },
    { key: 'body_side', label: 'Side Profile', icon: '│', required: true },
    { key: 'body_posterior', label: 'Posterior', icon: '◎', description: 'Back view', required: true }
];
