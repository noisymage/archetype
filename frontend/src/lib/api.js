/**
 * API client for backend communication
 */

const API_BASE = '';

/**
 * Generic fetch wrapper with error handling
 */
async function apiFetch(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const response = await fetch(url, {
        headers: {
            'Content-Type': 'application/json',
            ...options.headers
        },
        ...options
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || 'API request failed');
    }

    return response.json();
}

// === Projects ===

export async function fetchProjects() {
    return apiFetch('/api/projects');
}

export async function createProject(name, loraPresetType = 'SDXL') {
    return apiFetch('/api/projects', {
        method: 'POST',
        body: JSON.stringify({ name, lora_preset_type: loraPresetType })
    });
}

export async function deleteProject(projectId) {
    return apiFetch(`/api/projects/${projectId}`, { method: 'DELETE' });
}

// === Characters ===

export async function fetchCharacters(projectId) {
    return apiFetch(`/api/projects/${projectId}/characters`);
}

export async function createCharacter(projectId, name, gender = 'neutral') {
    return apiFetch(`/api/projects/${projectId}/characters`, {
        method: 'POST',
        body: JSON.stringify({ name, gender })
    });
}

export async function getCharacter(characterId) {
    return apiFetch(`/api/characters/${characterId}`);
}

export async function deleteCharacter(characterId) {
    return apiFetch(`/api/characters/${characterId}`, { method: 'DELETE' });
}

// === Reference Images ===

export async function setReferenceImages(characterId, images, sourcePath = null) {
    return apiFetch(`/api/characters/${characterId}/references`, {
        method: 'POST',
        body: JSON.stringify({ images, source_path: sourcePath })
    });
}

export async function getReferenceImages(characterId) {
    return apiFetch(`/api/characters/${characterId}/references`);
}

// === Dataset Images ===

export async function fetchDatasetImages(characterId) {
    return apiFetch(`/api/characters/${characterId}/images`);
}

// === Folder Scanning ===

export async function scanFolder(folderPath, characterId) {
    return apiFetch('/api/scan-folder', {
        method: 'POST',
        body: JSON.stringify({ folder_path: folderPath, character_id: characterId })
    });
}

export async function listImages(folderPath) {
    return apiFetch(`/api/list-images?folder_path=${encodeURIComponent(folderPath)}`);
}

// === Batch Processing ===

export async function startBatchProcess(characterId, reprocessAll = false) {
    return apiFetch('/api/process/batch', {
        method: 'POST',
        body: JSON.stringify({ character_id: characterId, reprocess_all: reprocessAll })
    });
}

export async function getJobStatus(jobId) {
    return apiFetch(`/api/process/status/${jobId}`);
}

export async function cancelJob(jobId) {
    return apiFetch(`/api/process/${jobId}/cancel`, { method: 'POST' });
}

export async function getCharacterReferences(characterId) {
    const res = await fetch(`${API_BASE}/api/characters/${characterId}/references`);
    if (!res.ok) throw new Error('Failed to get references');
    return res.json();
}

export async function setCharacterReferences(characterId, imagePaths, sourcePath = null) {
    const res = await fetch(`${API_BASE}/api/characters/${characterId}/references`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ images: imagePaths, source_path: sourcePath })
    });
    if (!res.ok) throw new Error('Failed to set references');
    return res.json();
}

// === Reference Analysis ===

export async function analyzeReferences(imagePaths, gender = 'neutral') {
    const res = await fetch(`${API_BASE}/api/analyze/reference`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ images: imagePaths, gender })
    });
    if (!res.ok) throw new Error('Failed to analyze references');
    return res.json();
}

export async function reprocessImage(imageId) {
    return apiFetch(`/api/images/${imageId}/reprocess`, {
        method: 'POST'
    });
}

// === Image URLs ===

export function getThumbnailUrl(imagePath, size = 256) {
    return `${API_BASE}/api/image/thumbnail?path=${encodeURIComponent(imagePath)}&size=${size}`;
}

export function getFullImageUrl(imagePath) {
    return `${API_BASE}/api/image/serve?path=${encodeURIComponent(imagePath)}`;
}

// === WebSocket ===

export function createProgressWebSocket(jobId, onMessage) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/process/${jobId}`);

    ws.onopen = () => {
        console.log(`WebSocket connected for job ${jobId}`);
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            onMessage(data);
        } catch (e) {
            // Handle non-JSON messages (like pong)
            console.log('WebSocket message:', event.data);
        }
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
        console.log(`WebSocket closed for job ${jobId}`);
    };

    // Keep-alive ping
    const pingInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send('ping');
        }
    }, 30000);

    // Return cleanup function
    const close = () => {
        clearInterval(pingInterval);
        if (ws.readyState === WebSocket.OPEN) {
            ws.close();
        }
    };

    return { ws, close };
}

// === Health Check ===

export async function checkHealth() {
    try {
        const result = await apiFetch('/api/health');
        return result.status === 'ok';
    } catch {
        return false;
    }
}

// === Captions ===

export async function getImageCaptions(imageId) {
    return apiFetch(`/api/images/${imageId}/captions`);
}

export async function updateCaption(captionId, textContent) {
    return apiFetch(`/api/captions/${captionId}`, {
        method: 'PUT',
        body: JSON.stringify({ text_content: textContent })
    });
}
