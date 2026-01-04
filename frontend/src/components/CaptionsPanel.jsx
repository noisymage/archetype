import { useState, useEffect } from 'react';
import { Edit2, Save, X, Loader2, FileText, Copy, Check } from 'lucide-react';
import { cn } from '../lib/utils';
import { getImageCaptions, updateCaption } from '../lib/api';

/**
 * CaptionsPanel - Displays and allows editing of image captions
 */
export function CaptionsPanel({ imageId }) {
    const [captions, setCaptions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [editingId, setEditingId] = useState(null);
    const [editText, setEditText] = useState('');
    const [saving, setSaving] = useState(false);
    const [copiedId, setCopiedId] = useState(null);

    // Fetch captions when image changes
    useEffect(() => {
        if (!imageId) return;

        setLoading(true);
        getImageCaptions(imageId)
            .then(data => setCaptions(data))
            .catch(err => console.error('Failed to load captions:', err))
            .finally(() => setLoading(false));
    }, [imageId]);

    const handleEdit = (caption) => {
        setEditingId(caption.id);
        setEditText(caption.text_content);
    };

    const handleCancel = () => {
        setEditingId(null);
        setEditText('');
    };

    const handleSave = async (captionId) => {
        setSaving(true);
        try {
            const updated = await updateCaption(captionId, editText);
            setCaptions(prev =>
                prev.map(c => c.id === captionId ? updated : c)
            );
            setEditingId(null);
            setEditText('');
        } catch (err) {
            console.error('Failed to save caption:', err);
            alert('Failed to save caption');
        } finally {
            setSaving(false);
        }
    };

    const handleCopy = async (caption) => {
        try {
            await navigator.clipboard.writeText(caption.text_content);
            setCopiedId(caption.id);
            setTimeout(() => setCopiedId(null), 2000);
        } catch (err) {
            console.error('Failed to copy:', err);
        }
    };

    // Format type label for display
    const formatLabel = (type) => {
        switch (type) {
            case 'SDXL': return 'SDXL';
            case 'Flux': return 'Flux';
            case 'Qwen-Image': return 'Qwen';
            case 'Z-Image': return 'Z-Img';
            default: return type;
        }
    };

    const formatColor = (type) => {
        switch (type) {
            case 'SDXL': return 'text-cyan-400 bg-cyan-500/10 border-cyan-500/30';
            case 'Flux': return 'text-fuchsia-400 bg-fuchsia-500/10 border-fuchsia-500/30';
            case 'Qwen-Image': return 'text-amber-400 bg-amber-500/10 border-amber-500/30';
            case 'Z-Image': return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30';
            default: return 'text-zinc-400 bg-zinc-500/10 border-zinc-500/30';
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center py-8">
                <Loader2 className="w-5 h-5 text-zinc-500 animate-spin" />
            </div>
        );
    }

    if (captions.length === 0) {
        return (
            <div className="text-center py-6">
                <FileText className="w-8 h-8 text-zinc-700 mx-auto mb-2" />
                <p className="text-xs text-zinc-500">No captions generated</p>
                <p className="text-[10px] text-zinc-600 mt-1">Process with LLM enrichment to generate</p>
            </div>
        );
    }

    return (
        <div className="space-y-3">
            {captions.map(caption => (
                <div
                    key={caption.id}
                    className="bg-zinc-900/50 rounded-lg border border-white/5 overflow-hidden"
                >
                    {/* Header */}
                    <div className="flex items-center justify-between px-3 py-2 border-b border-white/5">
                        <span className={cn(
                            "text-[10px] font-medium uppercase tracking-wider px-2 py-0.5 rounded border",
                            formatColor(caption.model_type)
                        )}>
                            {formatLabel(caption.model_type)}
                        </span>

                        <div className="flex items-center gap-1">
                            {editingId !== caption.id && (
                                <>
                                    <button
                                        onClick={() => handleCopy(caption)}
                                        className="p-1 rounded hover:bg-white/10 text-zinc-500 hover:text-zinc-300 transition-colors"
                                        title="Copy caption"
                                    >
                                        {copiedId === caption.id ? (
                                            <Check className="w-3 h-3 text-green-400" />
                                        ) : (
                                            <Copy className="w-3 h-3" />
                                        )}
                                    </button>
                                    <button
                                        onClick={() => handleEdit(caption)}
                                        className="p-1 rounded hover:bg-white/10 text-zinc-500 hover:text-zinc-300 transition-colors"
                                        title="Edit caption"
                                    >
                                        <Edit2 className="w-3 h-3" />
                                    </button>
                                </>
                            )}
                        </div>
                    </div>

                    {/* Content */}
                    <div className="p-3">
                        {editingId === caption.id ? (
                            <div className="space-y-2">
                                <textarea
                                    value={editText}
                                    onChange={(e) => setEditText(e.target.value)}
                                    className="w-full bg-zinc-800 border border-white/10 rounded-md px-3 py-2 text-xs text-zinc-200 focus:outline-none focus:border-cyan-500/50 resize-none"
                                    rows={4}
                                    autoFocus
                                />
                                <div className="flex justify-end gap-2">
                                    <button
                                        onClick={handleCancel}
                                        disabled={saving}
                                        className="px-2 py-1 text-xs text-zinc-400 hover:text-zinc-200 transition-colors"
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        onClick={() => handleSave(caption.id)}
                                        disabled={saving}
                                        className="flex items-center gap-1 px-3 py-1 text-xs bg-cyan-500/20 text-cyan-400 rounded hover:bg-cyan-500/30 transition-colors disabled:opacity-50"
                                    >
                                        {saving ? (
                                            <Loader2 className="w-3 h-3 animate-spin" />
                                        ) : (
                                            <Save className="w-3 h-3" />
                                        )}
                                        Save
                                    </button>
                                </div>
                            </div>
                        ) : (
                            <p className="text-xs text-zinc-300 leading-relaxed whitespace-pre-wrap">
                                {caption.text_content || <span className="text-zinc-600 italic">Empty caption</span>}
                            </p>
                        )}
                    </div>
                </div>
            ))}
        </div>
    );
}
