import { useState, useEffect } from 'react';
import { X, Loader2, Check, AlertCircle, Server, Cloud } from 'lucide-react';
import { cn } from '../lib/utils';
import { Button } from './ui/Button';

/**
 * Settings modal for configuring LLM providers and processing options
 */
export function SettingsModal({ isOpen, onClose }) {
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState(null);

    // Settings state
    const [settings, setSettings] = useState({
        llm: {
            provider: 'ollama',
            ollama: { base_url: 'http://localhost:11434', model: 'llava:13b' },
            gemini: { configured: false, model: 'gemini-2.0-flash-exp' }
        },
        processing: { enable_enrichment: true, caption_formats: [] }
    });

    // LLM status
    const [llmStatus, setLlmStatus] = useState(null);

    // Form values (separate from persisted settings for API key handling)
    const [geminiApiKey, setGeminiApiKey] = useState('');

    useEffect(() => {
        if (isOpen) {
            loadSettings();
            loadLlmStatus();
        }
    }, [isOpen]);

    const loadSettings = async () => {
        try {
            setLoading(true);
            const res = await fetch('/api/settings');
            if (res.ok) {
                const data = await res.json();
                setSettings(data);
            }
        } catch (e) {
            setError('Failed to load settings');
        } finally {
            setLoading(false);
        }
    };

    const loadLlmStatus = async () => {
        try {
            const res = await fetch('/api/llm/status');
            if (res.ok) {
                const data = await res.json();
                setLlmStatus(data);
            }
        } catch (e) {
            console.error('Failed to load LLM status', e);
        }
    };

    const handleSave = async () => {
        try {
            setSaving(true);
            setError(null);

            const updates = {
                llm_provider: settings.llm.provider,
                ollama_base_url: settings.llm.ollama.base_url,
                ollama_model: settings.llm.ollama.model,
                gemini_model: settings.llm.gemini.model,
                enable_enrichment: settings.processing.enable_enrichment,
                caption_formats: settings.processing.caption_formats
            };

            // Only send API key if it was changed
            if (geminiApiKey) {
                updates.gemini_api_key = geminiApiKey;
            }

            const res = await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updates)
            });

            if (res.ok) {
                // Refresh LLM status after saving
                await loadLlmStatus();
                onClose();
            } else {
                setError('Failed to save settings');
            }
        } catch (e) {
            setError('Failed to save settings');
        } finally {
            setSaving(false);
        }
    };

    if (!isOpen) return null;

    const captionFormats = ['SDXL', 'Flux', 'Qwen-Image', 'Z-Image'];

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                onClick={onClose}
            />

            {/* Modal */}
            <div className="relative bg-zinc-900 border border-white/10 rounded-xl w-full max-w-lg mx-4 shadow-2xl">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-white/10">
                    <h2 className="text-lg font-semibold text-white">Settings</h2>
                    <button
                        onClick={onClose}
                        className="p-1.5 rounded-lg hover:bg-white/5 transition-colors text-zinc-400 hover:text-white"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Content */}
                <div className="p-4 space-y-6 max-h-[60vh] overflow-y-auto">
                    {loading ? (
                        <div className="flex items-center justify-center py-8">
                            <Loader2 className="w-6 h-6 animate-spin text-cyan-400" />
                        </div>
                    ) : (
                        <>
                            {/* LLM Provider Selection */}
                            <div className="space-y-3">
                                <label className="text-sm font-medium text-zinc-300">LLM Provider</label>
                                <div className="grid grid-cols-2 gap-3">
                                    {/* Ollama */}
                                    <button
                                        onClick={() => setSettings(s => ({
                                            ...s,
                                            llm: { ...s.llm, provider: 'ollama' }
                                        }))}
                                        className={cn(
                                            "p-4 rounded-lg border transition-all text-left",
                                            settings.llm.provider === 'ollama'
                                                ? "border-cyan-500/50 bg-cyan-500/10"
                                                : "border-white/10 hover:border-white/20 bg-white/5"
                                        )}
                                    >
                                        <div className="flex items-center gap-2 mb-2">
                                            <Server className="w-5 h-5 text-cyan-400" />
                                            <span className="font-medium text-white">Ollama</span>
                                        </div>
                                        <p className="text-xs text-zinc-400">Local inference</p>
                                        {llmStatus?.providers?.ollama && (
                                            <div className={cn(
                                                "mt-2 text-xs flex items-center gap-1",
                                                llmStatus.providers.ollama.available
                                                    ? "text-green-400"
                                                    : "text-amber-400"
                                            )}>
                                                {llmStatus.providers.ollama.available ? (
                                                    <><Check className="w-3 h-3" /> Available</>
                                                ) : (
                                                    <><AlertCircle className="w-3 h-3" /> Not running</>
                                                )}
                                            </div>
                                        )}
                                    </button>

                                    {/* Gemini */}
                                    <button
                                        onClick={() => setSettings(s => ({
                                            ...s,
                                            llm: { ...s.llm, provider: 'gemini' }
                                        }))}
                                        className={cn(
                                            "p-4 rounded-lg border transition-all text-left",
                                            settings.llm.provider === 'gemini'
                                                ? "border-fuchsia-500/50 bg-fuchsia-500/10"
                                                : "border-white/10 hover:border-white/20 bg-white/5"
                                        )}
                                    >
                                        <div className="flex items-center gap-2 mb-2">
                                            <Cloud className="w-5 h-5 text-fuchsia-400" />
                                            <span className="font-medium text-white">Gemini</span>
                                        </div>
                                        <p className="text-xs text-zinc-400">Cloud inference</p>
                                        {llmStatus?.providers?.gemini && (
                                            <div className={cn(
                                                "mt-2 text-xs flex items-center gap-1",
                                                llmStatus.providers.gemini.available
                                                    ? "text-green-400"
                                                    : llmStatus.providers.gemini.configured
                                                        ? "text-amber-400"
                                                        : "text-zinc-500"
                                            )}>
                                                {llmStatus.providers.gemini.available ? (
                                                    <><Check className="w-3 h-3" /> Connected</>
                                                ) : llmStatus.providers.gemini.configured ? (
                                                    <><AlertCircle className="w-3 h-3" /> API error</>
                                                ) : (
                                                    <><AlertCircle className="w-3 h-3" /> Not configured</>
                                                )}
                                            </div>
                                        )}
                                    </button>
                                </div>
                            </div>

                            {/* Provider-specific settings */}
                            {settings.llm.provider === 'ollama' ? (
                                <div className="space-y-3 p-4 rounded-lg bg-white/5 border border-white/10">
                                    <h3 className="text-sm font-medium text-zinc-300">Ollama Settings</h3>
                                    <div className="space-y-2">
                                        <label className="text-xs text-zinc-400">Base URL</label>
                                        <input
                                            type="text"
                                            value={settings.llm.ollama.base_url}
                                            onChange={(e) => setSettings(s => ({
                                                ...s,
                                                llm: { ...s.llm, ollama: { ...s.llm.ollama, base_url: e.target.value } }
                                            }))}
                                            className="w-full px-3 py-2 bg-black/30 border border-white/10 rounded-lg text-sm text-white focus:border-cyan-500/50 focus:outline-none"
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <label className="text-xs text-zinc-400">Model</label>
                                        <input
                                            type="text"
                                            value={settings.llm.ollama.model}
                                            onChange={(e) => setSettings(s => ({
                                                ...s,
                                                llm: { ...s.llm, ollama: { ...s.llm.ollama, model: e.target.value } }
                                            }))}
                                            className="w-full px-3 py-2 bg-black/30 border border-white/10 rounded-lg text-sm text-white focus:border-cyan-500/50 focus:outline-none"
                                            placeholder="llava:13b"
                                        />
                                    </div>
                                </div>
                            ) : (
                                <div className="space-y-3 p-4 rounded-lg bg-white/5 border border-white/10">
                                    <h3 className="text-sm font-medium text-zinc-300">Gemini Settings</h3>
                                    <div className="space-y-2">
                                        <label className="text-xs text-zinc-400">API Key</label>
                                        <input
                                            type="password"
                                            value={geminiApiKey}
                                            onChange={(e) => setGeminiApiKey(e.target.value)}
                                            placeholder={settings.llm.gemini.configured ? "••••••••" : "Enter API key"}
                                            className="w-full px-3 py-2 bg-black/30 border border-white/10 rounded-lg text-sm text-white focus:border-fuchsia-500/50 focus:outline-none"
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <label className="text-xs text-zinc-400">Model</label>
                                        <select
                                            value={settings.llm.gemini.model}
                                            onChange={(e) => setSettings(s => ({
                                                ...s,
                                                llm: { ...s.llm, gemini: { ...s.llm.gemini, model: e.target.value } }
                                            }))}
                                            className="w-full px-3 py-2 bg-black/30 border border-white/10 rounded-lg text-sm text-white focus:border-fuchsia-500/50 focus:outline-none"
                                        >
                                            <option value="gemini-3-flash-preview">Gemini 3 Flash (Preview)</option>
                                            <option value="gemini-2.0-flash-exp">Gemini 2.0 Flash (Experimental)</option>
                                            <option value="gemini-1.5-flash">Gemini 1.5 Flash</option>
                                            <option value="gemini-1.5-pro">Gemini 1.5 Pro</option>
                                        </select>
                                    </div>
                                </div>
                            )}

                            {/* Caption Formats */}
                            <div className="space-y-3">
                                <label className="text-sm font-medium text-zinc-300">Caption Formats</label>
                                <div className="flex flex-wrap gap-2">
                                    {captionFormats.map(format => (
                                        <button
                                            key={format}
                                            onClick={() => {
                                                const current = settings.processing.caption_formats || [];
                                                const updated = current.includes(format)
                                                    ? current.filter(f => f !== format)
                                                    : [...current, format];
                                                setSettings(s => ({
                                                    ...s,
                                                    processing: { ...s.processing, caption_formats: updated }
                                                }));
                                            }}
                                            className={cn(
                                                "px-3 py-1.5 rounded-full text-sm transition-all border",
                                                (settings.processing.caption_formats || []).includes(format)
                                                    ? "bg-cyan-500/20 border-cyan-500/50 text-cyan-300"
                                                    : "bg-white/5 border-white/10 text-zinc-400 hover:text-white"
                                            )}
                                        >
                                            {format}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Enrichment Toggle */}
                            <div className="flex items-center justify-between p-4 rounded-lg bg-white/5 border border-white/10">
                                <div>
                                    <p className="text-sm font-medium text-zinc-300">Enable LLM Enrichment</p>
                                    <p className="text-xs text-zinc-500">Generate descriptions and captions during processing</p>
                                </div>
                                <button
                                    onClick={() => setSettings(s => ({
                                        ...s,
                                        processing: { ...s.processing, enable_enrichment: !s.processing.enable_enrichment }
                                    }))}
                                    className={cn(
                                        "w-12 h-6 rounded-full transition-colors relative",
                                        settings.processing.enable_enrichment ? "bg-cyan-500" : "bg-zinc-700"
                                    )}
                                >
                                    <div className={cn(
                                        "absolute top-1 w-4 h-4 rounded-full bg-white transition-all",
                                        settings.processing.enable_enrichment ? "left-7" : "left-1"
                                    )} />
                                </button>
                            </div>
                        </>
                    )}

                    {error && (
                        <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
                            {error}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="flex items-center justify-end gap-3 p-4 border-t border-white/10">
                    <Button variant="ghost" onClick={onClose}>
                        Cancel
                    </Button>
                    <Button variant="primary" onClick={handleSave} disabled={saving}>
                        {saving ? (
                            <><Loader2 className="w-4 h-4 animate-spin mr-2" /> Saving...</>
                        ) : (
                            'Save Settings'
                        )}
                    </Button>
                </div>
            </div>
        </div>
    );
}
