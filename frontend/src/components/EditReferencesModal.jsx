import { useState, useEffect } from 'react';
import { X, Search, Loader2, AlertTriangle, CheckCircle, Pencil } from 'lucide-react';
import { cn } from '../lib/utils';
import { Button } from './ui/Button';
import * as api from '../lib/api';
import { getThumbnailUrl } from '../lib/api';

// Import slot definitions from CreateProjectWizard
const headSlots = [
    { key: 'head_front', label: 'Front', icon: '⚫', description: 'Face camera', required: true },
    { key: 'head_45l', label: '45° Left', icon: '←', description: "Viewer's left", required: true },
    { key: 'head_45r', label: '45° Right', icon: '→', description: "Viewer's right", required: true },
    { key: 'head_90l', label: '90° Left Profile', icon: '⮜', description: 'Full left side', required: false, optional: true },
    { key: 'head_90r', label: '90° Right Profile', icon: '⮞', description: 'Full right side', required: false, optional: true },
    { key: 'head_up', label: 'Looking Up', icon: '⮝', description: 'Pitch +30°', required: false, optional: true },
    { key: 'head_down', label: 'Looking Down', icon: '⮟', description: 'Pitch -30°', required: false, optional: true },
];

const bodySlots = [
    { key: 'body_front', label: 'A-Pose Front', icon: '╋', required: true },
    { key: 'body_side', label: 'Side Profile', icon: '│', required: true },
    { key: 'body_posterior', label: 'Posterior', icon: '◎', description: 'Back view', required: true }
];

/**
 * Reference slot component for edit modal
 */
function ReferenceSlotEdit({ slot, currentImage, selectedImage, onSelect, optional = false }) {
    const hasImage = currentImage || selectedImage;
    const isReplacing = currentImage && selectedImage && currentImage.path !== selectedImage.path;

    return (
        <div
            onClick={onSelect}
            className={cn(
                "relative border rounded-lg p-3 cursor-pointer transition-all overflow-hidden group",
                hasImage ? "border-cyan-500/50 bg-cyan-500/10" : "border-white/10 hover:border-white/20",
                isReplacing && "border-yellow-500/50 bg-yellow-500/10",
                optional && "opacity-60 hover:opacity-100"
            )}
        >
            {/* Background Thumbnail (Blurred) if image exists */}
            {hasImage && (
                <div className="absolute inset-0 z-0">
                    <img
                        src={getThumbnailUrl(selectedImage || (currentImage?.path))}
                        alt="bg"
                        className="w-full h-full object-cover opacity-20 group-hover:opacity-30 transition-opacity"
                    />
                </div>
            )}

            <div className="relative z-10">
                <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                        <span className="text-2xl drop-shadow-md">{slot.icon}</span>
                        <div>
                            <div className="text-sm font-medium text-white drop-shadow-md">{slot.label}</div>
                            {slot.description && (
                                <div className="text-xs text-zinc-400 drop-shadow-md">{slot.description}</div>
                            )}
                        </div>
                    </div>
                    {!slot.required && <span className="text-xs text-zinc-500 bg-black/50 px-1.5 py-0.5 rounded">Optional</span>}
                </div>

                {hasImage && (
                    <div className="mt-2 aspect-square w-full rounded-md overflow-hidden bg-black/50 border border-white/10 relative">
                        <img
                            src={getThumbnailUrl(selectedImage || (currentImage?.path))}
                            alt={slot.label}
                            className="w-full h-full object-cover"
                        />
                        {isReplacing && (
                            <div className="absolute top-0 right-0 bg-yellow-500 text-black text-[10px] font-bold px-1.5 py-0.5 rounded-bl">
                                NEW
                            </div>
                        )}
                    </div>
                )}

                {hasImage ? (
                    <div className="text-[10px] text-zinc-400 mt-2 truncate font-mono bg-black/50 p-1 rounded">
                        {(selectedImage || currentImage?.path).split('/').pop()}
                    </div>
                ) : (
                    <div className="text-xs text-zinc-500 mt-4 text-center py-4 border-2 border-dashed border-white/5 rounded">
                        Click to assign
                    </div>
                )}
            </div>
        </div>
    );
}

/**
 * Modal for editing character reference images
 */
export default function EditReferencesModal({ character, open, onClose, onSave }) {
    const [refFolderPath, setRefFolderPath] = useState('');
    const [availableImages, setAvailableImages] = useState([]);
    const [currentRefs, setCurrentRefs] = useState({});
    const [selectedRefs, setSelectedRefs] = useState({});
    const [isScanning, setIsScanning] = useState(false);
    const [isValidating, setIsValidating] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [validationResult, setValidationResult] = useState(null);
    const [error, setError] = useState(null);
    const [activeSlot, setActiveSlot] = useState(null);

    // Load current references on mount
    useEffect(() => {
        if (open && character) {
            loadCurrentReferences();
        }
    }, [open, character]);

    const loadCurrentReferences = async () => {
        try {
            const refs = await api.getCharacterReferences(character.id);
            const refMap = {};
            refs.forEach(ref => {
                refMap[ref.view_type] = ref;
            });
            setCurrentRefs(refMap);
        } catch (err) {
            setError(`Failed to load references: ${err.message}`);
        }
    };

    const handleScanFolder = async () => {
        setIsScanning(true);
        setError(null);
        try {
            const result = await api.listImages(refFolderPath);
            // Handle both array and object response (just in case)
            const images = Array.isArray(result) ? result : (result.images || []);
            setAvailableImages(images);

            if (!images || images.length === 0) {
                setError('No supported images found in folder');
            }
        } catch (err) {
            setError(`Scan failed: ${err.message}`);
        } finally {
            setIsScanning(false);
        }
    };

    const handleSelectImage = (imagePath) => {
        if (!activeSlot) return;

        setSelectedRefs(prev => ({
            ...prev,
            [activeSlot]: imagePath
        }));
        setActiveSlot(null);
        setValidationResult(null); // Clear validation when changing images
    };

    const handleValidate = async () => {
        setIsValidating(true);
        setError(null);
        try {
            // Build paths object with current + selected
            const paths = {};
            const allSlots = [...headSlots, ...bodySlots];

            for (const slot of allSlots) {
                const selected = selectedRefs[slot.key];
                const current = currentRefs[slot.key];

                if (selected) {
                    paths[slot.key] = selected;
                } else if (current) {
                    paths[slot.key] = current.path;
                }
            }

            const result = await api.analyzeReferences(paths, character.gender);
            setValidationResult(result);
        } catch (err) {
            setError(`Validation failed: ${err.message}`);
        } finally {
            setIsValidating(false);
        }
    };

    const handleSave = async () => {
        if (!validationResult || !validationResult.success) {
            setError('Please validate references before saving');
            return;
        }

        setIsSaving(true);
        setError(null);
        try {
            // Build final paths object
            const paths = {};
            const allSlots = [...headSlots, ...bodySlots];

            for (const slot of allSlots) {
                const selected = selectedRefs[slot.key];
                const current = currentRefs[slot.key];

                if (selected) {
                    paths[slot.key] = selected;
                } else if (current) {
                    paths[slot.key] = current.path;
                }
            }

            await api.setCharacterReferences(character.id, paths);
            onSave?.();
            onClose();
        } catch (err) {
            setError(`Save failed: ${err.message}`);
        } finally {
            setIsSaving(false);
        }
    };

    const hasChanges = Object.keys(selectedRefs).length > 0;
    const requiredSlots = [...headSlots.filter(s => s.required), ...bodySlots];
    const allRequiredFilled = requiredSlots.every(slot =>
        selectedRefs[slot.key] || currentRefs[slot.key]
    );

    if (!open) return null;

    return (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
            <div className="bg-zinc-900 rounded-xl border border-white/10 max-w-5xl w-full max-h-[90vh] overflow-y-auto">
                {/* Header */}
                <div className="sticky top-0 bg-zinc-900 border-b border-white/10 p-6 flex items-center justify-between">
                    <div>
                        <h2 className="text-2xl font-bold text-white">Edit Reference Images</h2>
                        <p className="text-sm text-zinc-400 mt-1">{character.name}</p>
                    </div>
                    <button
                        onClick={onClose}
                        className="text-zinc-400 hover:text-white transition-colors"
                    >
                        <X className="w-6 h-6" />
                    </button>
                </div>

                <div className="p-6 space-y-6">
                    {/* Error Display */}
                    {error && (
                        <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-4 flex items-start gap-3">
                            <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                            <p className="text-sm text-red-200">{error}</p>
                        </div>
                    )}

                    {/* Folder Scanner */}
                    <div>
                        <label className="block text-sm font-medium text-zinc-300 mb-2">
                            Reference Images Folder
                        </label>
                        <div className="flex gap-2">
                            <input
                                type="text"
                                value={refFolderPath}
                                onChange={(e) => setRefFolderPath(e.target.value)}
                                placeholder="/path/to/reference/images"
                                className="flex-1 px-3 py-2 bg-zinc-800 border border-white/10 rounded-lg text-white placeholder-zinc-500 focus:outline-none focus:border-cyan-500/50"
                            />
                            <Button
                                variant="secondary"
                                onClick={handleScanFolder}
                                disabled={isScanning || !refFolderPath.trim()}
                                className="gap-2"
                            >
                                {isScanning ? (
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                ) : (
                                    <Search className="w-4 h-4" />
                                )}
                                Scan
                            </Button>
                        </div>
                    </div>

                    {/* Available Images */}
                    {availableImages.length > 0 && (
                        <div>
                            <h3 className="text-sm font-medium text-zinc-300 mb-2">
                                Available Images ({availableImages.length})
                            </h3>
                            <div className="grid grid-cols-6 gap-2 bg-zinc-800 p-2 rounded-lg max-h-60 overflow-y-auto">
                                {availableImages.map((img, idx) => (
                                    <button
                                        key={idx}
                                        onClick={() => handleSelectImage(img)}
                                        className={cn(
                                            "group relative aspect-square rounded-md overflow-hidden border transition-all",
                                            activeSlot
                                                ? "border-cyan-500/50 hover:border-cyan-400 cursor-pointer"
                                                : "border-white/10 opacity-50 cursor-not-allowed"
                                        )}
                                        title={img.split('/').pop()}
                                        disabled={!activeSlot}
                                    >
                                        <img
                                            src={getThumbnailUrl(img)}
                                            alt="Thumbnail"
                                            className="w-full h-full object-cover transition-transform group-hover:scale-110"
                                        />
                                        <div className="absolute inset-x-0 bottom-0 bg-black/70 p-1">
                                            <p className="text-[10px] text-zinc-300 truncate font-mono text-center">
                                                {img.split('/').pop()}
                                            </p>
                                        </div>
                                        {activeSlot && (
                                            <div className="absolute inset-0 bg-cyan-500/20 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                                <span className="bg-black/70 text-white text-xs px-2 py-1 rounded">Assign</span>
                                            </div>
                                        )}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Reference Slots */}
                    <div className="space-y-6">
                        {/* Required Head References */}
                        <div>
                            <h3 className="text-sm font-medium text-zinc-300 mb-3">
                                Required Head References
                            </h3>
                            <div className="grid grid-cols-3 gap-3">
                                {headSlots.filter(s => s.required).map(slot => (
                                    <ReferenceSlotEdit
                                        key={slot.key}
                                        slot={slot}
                                        currentImage={currentRefs[slot.key]}
                                        selectedImage={selectedRefs[slot.key]}
                                        onSelect={() => setActiveSlot(slot.key)}
                                    />
                                ))}
                            </div>
                        </div>

                        {/* Optional Head References */}
                        <div>
                            <h3 className="text-xs font-medium text-zinc-500 mb-2">
                                Optional Head References (Improves Accuracy)
                            </h3>
                            <div className="grid grid-cols-4 gap-2">
                                {headSlots.filter(s => s.optional).map(slot => (
                                    <ReferenceSlotEdit
                                        key={slot.key}
                                        slot={slot}
                                        currentImage={currentRefs[slot.key]}
                                        selectedImage={selectedRefs[slot.key]}
                                        onSelect={() => setActiveSlot(slot.key)}
                                        optional={true}
                                    />
                                ))}
                            </div>
                        </div>

                        {/* Body References */}
                        <div>
                            <h3 className="text-sm font-medium text-zinc-300 mb-3">
                                Body References
                            </h3>
                            <div className="grid grid-cols-3 gap-3">
                                {bodySlots.map(slot => (
                                    <ReferenceSlotEdit
                                        key={slot.key}
                                        slot={slot}
                                        currentImage={currentRefs[slot.key]}
                                        selectedImage={selectedRefs[slot.key]}
                                        onSelect={() => setActiveSlot(slot.key)}
                                    />
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Validation */}
                    {hasChanges && (
                        <div className="bg-zinc-800 rounded-lg p-4">
                            <Button
                                onClick={handleValidate}
                                disabled={isValidating || !allRequiredFilled}
                                className="w-full gap-2"
                                variant="secondary"
                            >
                                {isValidating ? (
                                    <>
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        Validating...
                                    </>
                                ) : (
                                    <>
                                        <CheckCircle className="w-4 h-4" />
                                        Re-Validate References
                                    </>
                                )}
                            </Button>

                            {validationResult && (
                                <div className="mt-4">
                                    {validationResult.success ? (
                                        <div className="text-sm text-green-400 flex items-center gap-2">
                                            <CheckCircle className="w-4 h-4" />
                                            Validation successful!
                                        </div>
                                    ) : (
                                        <div className="text-sm text-red-400 flex items-start gap-2">
                                            <AlertTriangle className="w-4 h-4 mt-0.5" />
                                            <div>
                                                {validationResult.error || 'Validation failed'}
                                            </div>
                                        </div>
                                    )}

                                    {validationResult.warnings?.length > 0 && (
                                        <div className="mt-2 space-y-1">
                                            {validationResult.warnings.map((w, i) => (
                                                <div key={i} className="text-xs text-yellow-400">
                                                    ⚠ {w.message}
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="sticky bottom-0 bg-zinc-900 border-t border-white/10 p-6 flex gap-3 justify-end">
                    <Button variant="secondary" onClick={onClose}>
                        Cancel
                    </Button>
                    <Button
                        onClick={handleSave}
                        disabled={!hasChanges || !validationResult?.success || isSaving}
                        className="gap-2"
                    >
                        {isSaving ? (
                            <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                Saving...
                            </>
                        ) : (
                            <>
                                <CheckCircle className="w-4 h-4" />
                                Save Changes
                            </>
                        )}
                    </Button>
                </div>
            </div>
        </div>
    );
}
