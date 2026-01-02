import { useState, useEffect } from 'react';
import { X, Search, Loader2, AlertTriangle, CheckCircle, Trash2 } from 'lucide-react';
import { cn } from '../lib/utils';
import { Button } from './ui/Button';
import * as api from '../lib/api';
import { getThumbnailUrl } from '../lib/api';
import { HEAD_SLOTS, BODY_SLOTS } from '../lib/constants';

/**
 * Reference slot component for edit modal
 */
function ReferenceSlotEdit({ slot, currentImage, selectedImage, onAssign, onClear, isActiveTarget }) {
    const hasImage = currentImage || selectedImage;
    // It's replacing if:
    // 1. We have a current image AND a selected image (that isn't null/cleared)
    // 2. AND the selected image path is different from current
    // OR
    // 3. We have no current image BUT we have a selected image (new assignment to empty slot)
    const isNew = selectedImage && (!currentImage || currentImage.path !== selectedImage);
    const isAssigned = Object.keys(selectedImage || {}).length > 0;

    // Determine display image path
    const displayPath = selectedImage || currentImage?.path;

    return (
        <div
            onClick={onAssign}
            className={cn(
                "relative border rounded-lg p-2 cursor-pointer transition-all overflow-hidden group h-full flex flex-col",
                isActiveTarget ? "border-cyan-500/50 bg-cyan-500/10 hover:border-cyan-400 ring-1 ring-cyan-500/30" : "border-white/10 hover:border-white/20 bg-zinc-900",
                hasImage && !isActiveTarget && "border-green-500/30 bg-green-500/5",
                isNew && "border-yellow-500/50 bg-yellow-500/10",
                slot.optional && !hasImage && "opacity-60 hover:opacity-100"
            )}
        >
            <div className="flex items-center justify-between mb-1.5 min-h-[24px]">
                <div className="flex items-center gap-1.5 overflow-hidden">
                    <span className="text-sm shrink-0">{slot.icon}</span>
                    <span className={cn("text-[10px] font-medium truncate", hasImage ? "text-white" : "text-zinc-500")}>
                        {slot.label}
                    </span>
                </div>
                {hasImage && slot.optional && (
                    <button
                        onClick={(e) => {
                            e.stopPropagation();
                            onClear();
                        }}
                        className="p-1 hover:bg-black/50 rounded text-zinc-500 hover:text-red-400 transition-colors"
                        title="Clear slot"
                    >
                        <Trash2 className="w-3 h-3" />
                    </button>
                )}
            </div>

            <div className="flex-1 min-h-[80px] w-full rounded-md overflow-hidden bg-black/40 border border-white/5 relative flex items-center justify-center">
                {displayPath ? (
                    <div className="relative w-full h-full">
                        <img
                            src={getThumbnailUrl(displayPath)}
                            alt={slot.label}
                            className="w-full h-full object-contain"
                        />
                        {isNew && (
                            <div className="absolute top-0 right-0 bg-yellow-500 text-black text-[9px] font-bold px-1.5 py-0.5 rounded-bl">
                                NEW
                            </div>
                        )}
                        <div className="absolute inset-x-0 bottom-0 bg-black/70 p-1 opacity-0 group-hover:opacity-100 transition-opacity">
                            <p className="text-[9px] text-zinc-300 truncate font-mono text-center">
                                {displayPath.split('/').pop()}
                            </p>
                        </div>
                    </div>
                ) : (
                    <div className="text-zinc-700 text-[9px] text-center px-1">
                        {isActiveTarget ? (
                            <span className="text-cyan-500 animate-pulse">Click to Assign</span>
                        ) : (
                            "Empty"
                        )}
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
    const [refFolderPath, setRefFolderPath] = useState(character?.reference_images_path || '');
    const [availableImages, setAvailableImages] = useState([]);
    const [currentRefs, setCurrentRefs] = useState({});
    const [selectedRefs, setSelectedRefs] = useState({}); // Stores changes: { slotKey: imagePath }
    const [isScanning, setIsScanning] = useState(false);
    const [isValidating, setIsValidating] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [validationResult, setValidationResult] = useState(null);
    const [error, setError] = useState(null);
    const [selectedImageToAssign, setSelectedImageToAssign] = useState(null); // The image selected from gallery

    // Load current references on mount
    useEffect(() => {
        if (open && character) {
            setSelectedRefs({}); // Reset changes on open
            setValidationResult(null);
            setError(null);
            loadCurrentReferences();
            if (character.reference_images_path) {
                setRefFolderPath(character.reference_images_path);
                // Auto-scan if path exists
                handleScanFolder(character.reference_images_path, true);
            }
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

    const handleScanFolder = async (path = refFolderPath, silent = false) => {
        if (!path) return;
        setIsScanning(true);
        if (!silent) setError(null);
        try {
            const result = await api.listImages(path);
            // Handle both array and object response (just in case)
            const images = Array.isArray(result) ? result : (result.images || []);
            setAvailableImages(images);

            if ((!images || images.length === 0) && !silent) {
                setError('No supported images found in folder');
            }
        } catch (err) {
            if (!silent) setError(`Scan failed: ${err.message}`);
        } finally {
            setIsScanning(false);
        }
    };

    const handleSelectImageFromGallery = (imagePath) => {
        if (selectedImageToAssign === imagePath) {
            setSelectedImageToAssign(null); // Deselect
        } else {
            setSelectedImageToAssign(imagePath);
        }
    };

    const handleAssignSlot = (slotKey) => {
        if (selectedImageToAssign) {
            setSelectedRefs(prev => ({
                ...prev,
                [slotKey]: selectedImageToAssign
            }));
            setSelectedImageToAssign(null); // Clear selection after assignment
            setValidationResult(null); // Clear validation
        }
    };

    const handleClearSlot = (slotKey) => {
        setSelectedRefs(prev => {
            const next = { ...prev };
            // Set to null to explicitly mark as cleared/removed
            next[slotKey] = null;
            return next;
        });
        setValidationResult(null);
    };

    const handleValidate = async () => {
        setIsValidating(true);
        setError(null);
        try {
            const paths = buildFinalPaths();
            const result = await api.analyzeReferences(paths, character.gender);
            setValidationResult(result);
        } catch (err) {
            setError(`Validation failed: ${err.message}`);
        } finally {
            setIsValidating(false);
        }
    };

    const buildFinalPaths = () => {
        const paths = {};
        const allSlots = [...HEAD_SLOTS, ...BODY_SLOTS];

        for (const slot of allSlots) {
            const selected = selectedRefs[slot.key];
            const current = currentRefs[slot.key];

            if (selected === null) {
                // Explicitly cleared
                continue;
            } else if (selected) {
                // Newly assigned
                paths[slot.key] = selected;
            } else if (current) {
                // Kept existing
                paths[slot.key] = current.path;
            }
        }
        return paths;
    };

    const handleSave = async () => {
        if (!validationResult || !validationResult.success) {
            setError('Please validate references before saving');
            return;
        }

        setIsSaving(true);
        setError(null);
        try {
            const paths = buildFinalPaths();
            await api.setCharacterReferences(character.id, paths, refFolderPath);
            onSave?.();
            onClose(); // Auto-close on success
        } catch (err) {
            setError(`Save failed: ${err.message}`);
        } finally {
            setIsSaving(false);
        }
    };

    const hasChanges = Object.keys(selectedRefs).length > 0;
    const requiredSlots = [...HEAD_SLOTS.filter(s => s.required), ...BODY_SLOTS];

    // Check if all required slots have a value (either current or new, and not cleared)
    const allRequiredFilled = requiredSlots.every(slot => {
        const selected = selectedRefs[slot.key];
        const current = currentRefs[slot.key];
        // If selected is null, it's cleared. If selected is string, it's set. If undefined, fallback to current.
        return selected !== null && (selected || current);
    });

    if (!open) return null;

    // Helper to render head slots in grid
    const renderHeadGrid = () => {
        // Grid structure: 5 columns
        // Row 1: . UpL Up UpR .
        // Row 2: 90L 45L Front 45R 90R
        // Row 3: . DownL Down DownR .

        const gridMap = [
            [null, 'head_up_l', 'head_up', 'head_up_r', null],
            ['head_90l', 'head_45l', 'head_front', 'head_45r', 'head_90r'],
            [null, 'head_down_l', 'head_down', 'head_down_r', null]
        ];

        return (
            <div className="grid grid-cols-5 gap-2 w-full max-w-3xl mx-auto">
                {gridMap.flat().map((slotKey, idx) => {
                    const slot = HEAD_SLOTS.find(s => s.key === slotKey);
                    if (!slot) return <div key={idx} className="aspect-[4/5]" />; // Spacer

                    const selection = selectedRefs[slot.key];
                    // If explicitly cleared (null), don't show current image
                    const effectiveCurrent = (selection === null) ? null : currentRefs[slot.key];
                    // If new selection (string), show it
                    const displayedSelection = (typeof selection === 'string') ? selection : null;

                    return (
                        <div key={slot.key} className="aspect-[3/4]">
                            <ReferenceSlotEdit
                                slot={slot}
                                currentImage={effectiveCurrent}
                                selectedImage={displayedSelection}
                                onAssign={() => handleAssignSlot(slot.key)}
                                onClear={() => handleClearSlot(slot.key)}
                                isActiveTarget={!!selectedImageToAssign}
                            />
                        </div>
                    );
                })}
            </div>
        );
    };

    return (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4 backdrop-blur-sm">
            <div className="bg-zinc-950 rounded-xl border border-white/10 max-w-[90vw] w-full max-h-[90vh] overflow-hidden flex flex-col shadow-2xl">
                {/* Header */}
                <div className="bg-zinc-900/50 border-b border-white/5 p-4 flex items-center justify-between shrink-0">
                    <div>
                        <h2 className="text-xl font-bold text-white tracking-tight">Edit References</h2>
                        <p className="text-xs text-zinc-400 mt-0.5 font-mono">{character.name} <span className="text-zinc-600">|</span> {character.id}</p>
                    </div>
                    <button
                        onClick={onClose}
                        className="text-zinc-500 hover:text-white transition-colors p-2 hover:bg-white/5 rounded-lg"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                <div className="flex-1 overflow-hidden flex flex-col lg:flex-row">
                    {/* Left Panel: Validation & Controls */}
                    <div className="w-full lg:w-80 bg-zinc-900/30 border-r border-white/5 flex flex-col shrink-0 overflow-y-auto">
                        <div className="p-4 space-y-6">
                            {/* Validation Status */}
                            <div className="bg-zinc-900 rounded-lg p-4 border border-white/5">
                                <h3 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-3">Validation Status</h3>
                                {validationResult ? (
                                    <div className="space-y-3">
                                        <div className={cn("flex items-center gap-2 text-sm font-medium", validationResult.success ? "text-green-400" : "text-yellow-400")}>
                                            {validationResult.success ? <CheckCircle className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4" />}
                                            {validationResult.success ? "Ready to Save" : "Issues Found"}
                                        </div>
                                        {validationResult.warnings?.length > 0 && (
                                            <div className="space-y-1.5 max-h-40 overflow-y-auto pr-1">
                                                {validationResult.warnings.map((w, i) => (
                                                    <div key={i} className="text-[10px] bg-black/30 p-2 rounded border border-white/5 text-zinc-300 leading-tight">
                                                        <span className={cn("font-bold mr-1", w.severity === 'error' ? "text-red-400" : "text-yellow-400")}>
                                                            {w.severity === 'error' ? 'ERR' : 'WARN'}
                                                        </span>
                                                        {w.message}
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                ) : (
                                    <div className="text-sm text-zinc-500 italic text-center py-4">
                                        No validation run yet.
                                    </div>
                                )}

                                <Button
                                    onClick={handleValidate}
                                    disabled={isValidating || !allRequiredFilled}
                                    variant="secondary"
                                    className="w-full mt-4 text-xs h-8"
                                >
                                    {isValidating ? <Loader2 className="w-3 h-3 animate-spin mr-2" /> : <CheckCircle className="w-3 h-3 mr-2" />}
                                    Validate References
                                </Button>
                                {error && (
                                    <div className="mt-3 text-[10px] text-red-400 bg-red-950/20 p-2 rounded border border-red-500/20">
                                        {error}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Middle Area: Interactive Grid */}
                    <div className="flex-1 overflow-y-auto bg-zinc-950 p-6">
                        <div className="max-w-4xl mx-auto space-y-8">

                            {/* Head Section */}
                            <div>
                                <h3 className="text-sm font-medium text-zinc-400 mb-4 flex items-center gap-2">
                                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-500" />
                                    Head References
                                </h3>
                                {renderHeadGrid()}
                            </div>

                            {/* Body Section */}
                            <div>
                                <h3 className="text-sm font-medium text-zinc-400 mb-4 flex items-center gap-2">
                                    <span className="w-1.5 h-1.5 rounded-full bg-fuchsia-500" />
                                    Body References
                                </h3>
                                <div className="grid grid-cols-3 gap-4 max-w-2xl mx-auto">
                                    {BODY_SLOTS.map(slot => {
                                        const selection = selectedRefs[slot.key];
                                        // If explicitly cleared (null), don't show current image
                                        const effectiveCurrent = (selection === null) ? null : currentRefs[slot.key];
                                        // If new selection (string), show it
                                        const displayedSelection = (typeof selection === 'string') ? selection : null;

                                        return (
                                            <div key={slot.key} className="aspect-[3/5]">
                                                <ReferenceSlotEdit
                                                    slot={slot}
                                                    currentImage={effectiveCurrent}
                                                    selectedImage={displayedSelection}
                                                    onAssign={() => handleAssignSlot(slot.key)}
                                                    onClear={() => handleClearSlot(slot.key)}
                                                    isActiveTarget={!!selectedImageToAssign}
                                                />
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Right Panel: Gallery */}
                    <div className="w-full lg:w-72 bg-zinc-900 border-l border-white/5 flex flex-col shrink-0">
                        {/* Folder Input */}
                        <div className="p-3 border-b border-white/5 bg-zinc-900">
                            <label className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider mb-1.5 block">Source Folder</label>
                            <div className="flex gap-1.5">
                                <input
                                    type="text"
                                    value={refFolderPath}
                                    onChange={(e) => setRefFolderPath(e.target.value)}
                                    placeholder="/path/to/images"
                                    className="flex-1 px-2 py-1.5 bg-zinc-950 border border-white/10 rounded text-xs text-zinc-300 font-mono focus:outline-none focus:border-cyan-500/50"
                                />
                                <button
                                    onClick={() => handleScanFolder()}
                                    disabled={isScanning || !refFolderPath.trim()}
                                    className="px-2 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded transition-colors disabled:opacity-50"
                                >
                                    {isScanning ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Search className="w-3.5 h-3.5" />}
                                </button>
                            </div>
                        </div>

                        {/* Gallery Grid */}
                        <div className="flex-1 overflow-y-auto p-3">
                            <div className="mb-2 flex items-center justify-between">
                                <span className="text-xs text-zinc-400 font-medium">Available Images</span>
                                <span className="text-[10px] text-zinc-600 bg-zinc-900 px-1.5 py-0.5 rounded border border-white/5">{availableImages.length} found</span>
                            </div>

                            {availableImages.length > 0 ? (
                                <div className="grid grid-cols-3 gap-2">
                                    {availableImages.map((img, idx) => (
                                        <button
                                            key={idx}
                                            onClick={() => handleSelectImageFromGallery(img)}
                                            className={cn(
                                                "aspect-square rounded overflow-hidden border transition-all relative group",
                                                selectedImageToAssign === img
                                                    ? "border-cyan-500 ring-2 ring-cyan-500/30 z-10"
                                                    : "border-white/10 hover:border-white/30"
                                            )}
                                        >
                                            <img
                                                src={getThumbnailUrl(img)}
                                                alt="thumb"
                                                className="w-full h-full object-cover"
                                            />
                                            {selectedImageToAssign === img && (
                                                <div className="absolute inset-0 bg-cyan-500/20 flex items-center justify-center">
                                                    <div className="w-4 h-4 bg-cyan-500 rounded-full flex items-center justify-center shadow-lg">
                                                        <CheckCircle className="w-3 h-3 text-black" />
                                                    </div>
                                                </div>
                                            )}
                                        </button>
                                    ))}
                                </div>
                            ) : (
                                <div className="h-32 flex flex-col items-center justify-center text-zinc-600 text-xs border border-dashed border-white/5 rounded-lg bg-zinc-900/50">
                                    <span>No images found</span>
                                </div>
                            )}
                        </div>

                        {/* Footer Actions */}
                        <div className="p-4 border-t border-white/5 bg-zinc-900 mt-auto">
                            {selectedImageToAssign && (
                                <div className="mb-3 text-xs bg-cyan-950/30 border border-cyan-500/20 text-cyan-200 p-2 rounded flex items-center justify-between">
                                    <span>Select a slot to assign</span>
                                    <button onClick={() => setSelectedImageToAssign(null)} className="hover:text-white"><X className="w-3 h-3" /></button>
                                </div>
                            )}
                            <div className="flex gap-2 justify-end">
                                <Button variant="secondary" onClick={onClose} size="sm" className="text-xs">
                                    Cancel
                                </Button>
                                <Button
                                    onClick={handleSave}
                                    disabled={!hasChanges || !validationResult?.success || isSaving}
                                    variant="primary"
                                    size="sm"
                                    className="text-xs"
                                >
                                    {isSaving ? (
                                        <>
                                            <Loader2 className="w-3 h-3 animate-spin mr-1.5" />
                                            Saving
                                        </>
                                    ) : (
                                        <>
                                            <CheckCircle className="w-3 h-3 mr-1.5" />
                                            Save Changes
                                        </>
                                    )}
                                </Button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
