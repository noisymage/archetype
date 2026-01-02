import { useState, useRef } from 'react';
import { X, FolderOpen, User, ChevronRight, ChevronLeft, Check, Upload, Loader2, AlertTriangle, Search, Trash2 } from 'lucide-react';
import { cn } from '../lib/utils';
import { Button } from './ui/Button';
import { useProject } from '../context/ProjectContext';
import * as api from '../lib/api';
import { HEAD_SLOTS, BODY_SLOTS } from '../lib/constants';

const STEPS = ['Project', 'Character', 'References', 'Dataset', 'Validate'];

/**
 * Create Project Wizard - Multi-step flow for project creation
 */
export function CreateProjectWizard({ isOpen, onClose, initialProject = null }) {
    const { createProject, createCharacter, setReferenceImages, scanFolder, analyzeReferences } = useProject();

    // Start at step 1 (Character) if adding to existing project, else 0 (Project)
    const [step, setStep] = useState(initialProject ? 1 : 0);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    // Form state
    const [projectName, setProjectName] = useState('');
    const [loraPreset, setLoraPreset] = useState('SDXL');
    const [characterName, setCharacterName] = useState('');
    const [gender, setGender] = useState('neutral');
    const [referenceImages, setReferenceImagesState] = useState({});
    const [folderPath, setFolderPath] = useState('');
    const [scanResult, setScanResult] = useState(null);
    const [validationResult, setValidationResult] = useState(null);

    // Reference picker state
    const [refFolderPath, setRefFolderPath] = useState('');
    const [availableImages, setAvailableImages] = useState([]);
    const [selectedImage, setSelectedImage] = useState(null);
    const [isScanning, setIsScanning] = useState(false);

    // Created entities
    const [createdProject, setCreatedProject] = useState(initialProject);
    const [createdCharacter, setCreatedCharacter] = useState(null);

    if (!isOpen) return null;

    const handleScanRefFolder = async () => {
        if (!refFolderPath.trim()) return;

        setIsScanning(true);
        setError(null);
        try {
            const result = await api.listImages(refFolderPath.trim());
            const images = Array.isArray(result) ? result : (result.images || []);
            setAvailableImages(images);
            if (images.length === 0) {
                setError('No images found in this folder');
            }
        } catch (err) {
            setError(err.message);
            setAvailableImages([]);
        } finally {
            setIsScanning(false);
        }
    };

    const handleAssignToSlot = (slotKey) => {
        if (selectedImage) {
            setReferenceImagesState(prev => ({
                ...prev,
                [slotKey]: { path: selectedImage }
            }));
            setSelectedImage(null);
        }
    };

    const handleNext = async () => {
        setError(null);
        setIsLoading(true);

        try {
            switch (step) {
                case 0: // Project step
                    if (!projectName.trim()) {
                        throw new Error('Project name is required');
                    }
                    const project = await createProject(projectName.trim(), loraPreset);
                    setCreatedProject(project);
                    break;

                case 1: // Character step
                    if (!characterName.trim()) {
                        throw new Error('Character name is required');
                    }
                    const character = await createCharacter(createdProject.id, characterName.trim(), gender);
                    setCreatedCharacter(character);
                    break;

                case 2: // References step
                    const requiredSlots = [...HEAD_SLOTS.filter(s => s.required), ...BODY_SLOTS.filter(s => s.required)];
                    const filledRequired = requiredSlots.filter(s => referenceImages[s.key]);
                    if (filledRequired.length < requiredSlots.length) {
                        throw new Error(`Please assign all ${requiredSlots.length} required reference images (${filledRequired.length}/${requiredSlots.length} filled)`);
                    }
                    // Save reference paths
                    const allSlots = [...HEAD_SLOTS, ...BODY_SLOTS];
                    const paths = {};
                    for (const slot of allSlots) {
                        if (referenceImages[slot.key]) { // Only save if an image is assigned
                            paths[slot.key] = referenceImages[slot.key].path;
                        }
                    }
                    await setReferenceImages(createdCharacter.id, paths, refFolderPath);
                    break;

                case 3: // Dataset step
                    if (!folderPath.trim()) {
                        throw new Error('Dataset folder path is required');
                    }
                    const result = await scanFolder(folderPath.trim(), createdCharacter.id);
                    setScanResult(result);
                    break;

                case 4: // Validate step
                    if (!validationResult) {
                        // Run validation
                        const paths = {};
                        for (const slot of [...HEAD_SLOTS, ...BODY_SLOTS]) {
                            if (referenceImages[slot.key]) {
                                paths[slot.key] = referenceImages[slot.key].path;
                            }
                        }
                        const validation = await analyzeReferences(paths, gender);
                        setValidationResult(validation);
                    }
                    // Complete wizard
                    onClose();
                    return;
            }

            setStep(prev => prev + 1);
        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    const handleBack = () => {
        setError(null);
        setStep(prev => Math.max(initialProject ? 1 : 0, prev - 1));
    };

    const handleRunValidation = async () => {
        setIsLoading(true);
        setError(null);
        try {
            const paths = {};
            for (const slot of [...HEAD_SLOTS, ...BODY_SLOTS]) {
                if (referenceImages[slot.key]) {
                    paths[slot.key] = referenceImages[slot.key].path;
                }
            }
            const validation = await analyzeReferences(paths, gender);
            setValidationResult(validation);
        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    const renderSlotButton = (slot) => {
        const assigned = referenceImages[slot.key];
        const isTarget = selectedImage && !assigned;
        const hasImage = assigned;
        const isActiveTarget = selectedImage;

        return (
            <div
                onClick={() => handleAssignToSlot(slot.key)}
                className={cn(
                    "relative border rounded-lg p-2 cursor-pointer transition-all overflow-hidden group h-full flex flex-col",
                    isActiveTarget ? "hover:border-cyan-400 ring-1 ring-cyan-500/30" : "hover:border-white/20",
                    // Target state (pulse if selected)
                    isActiveTarget && !hasImage ? "border-cyan-500/50 bg-cyan-500/10 animate-pulse" : "border-white/10 bg-zinc-900",
                    // Assigned state
                    hasImage ? "border-green-500/30 bg-green-500/5" : "",
                    // Optional empty state
                    slot.optional && !hasImage && !isActiveTarget && "opacity-60 hover:opacity-100"
                )}
            >
                <div className="flex items-center justify-between mb-1.5 min-h-[20px]">
                    <div className="flex items-center gap-1.5 overflow-hidden">
                        <span className="text-xs shrink-0">{slot.icon}</span>
                        <span className={cn("text-[9px] font-medium truncate", hasImage ? "text-white" : "text-zinc-500")}>
                            {slot.label}
                        </span>
                    </div>
                    {hasImage && slot.optional && (
                        <button
                            onClick={(e) => {
                                e.stopPropagation();
                                setReferenceImagesState(prev => {
                                    const next = { ...prev };
                                    delete next[slot.key];
                                    return next;
                                });
                            }}
                            className="p-1 hover:bg-black/50 rounded text-zinc-500 hover:text-red-400 transition-colors"
                            title="Clear slot"
                        >
                            <Trash2 className="w-3 h-3" />
                        </button>
                    )}
                </div>

                <div className="flex-1 min-h-[70px] w-full rounded-md overflow-hidden bg-black/40 border border-white/5 relative flex items-center justify-center">
                    {assigned ? (
                        <div className="relative w-full h-full">
                            <img
                                src={api.getThumbnailUrl(assigned.path, 128)}
                                alt={slot.label}
                                className="w-full h-full object-contain"
                            />
                            <div className="absolute inset-x-0 bottom-0 bg-black/70 p-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                <p className="text-[8px] text-zinc-300 truncate font-mono text-center">
                                    {assigned.path.split('/').pop()}
                                </p>
                            </div>
                        </div>
                    ) : (
                        <div className="text-zinc-700 text-[8px] text-center px-1">
                            {isActiveTarget ? (
                                <span className="text-cyan-500">Click to Assign</span>
                            ) : (
                                "Empty"
                            )}
                        </div>
                    )}
                </div>
            </div>
        );
    };

    const renderHeadGrid = () => {
        const gridMap = [
            [null, 'head_up_l', 'head_up', 'head_up_r', null],
            ['head_90l', 'head_45l', 'head_front', 'head_45r', 'head_90r'],
            [null, 'head_down_l', 'head_down', 'head_down_r', null]
        ];

        return (
            <div className="grid grid-cols-5 gap-2 w-full max-w-xl mx-auto">
                {gridMap.flat().map((slotKey, idx) => {
                    const slot = HEAD_SLOTS.find(s => s.key === slotKey);
                    if (!slot) return <div key={idx} className="aspect-[3/4]" />; // Spacer

                    return (
                        <div key={slot.key} className="aspect-[3/4]">
                            {renderSlotButton(slot)}
                        </div>
                    );
                })}
            </div>
        );
    }

    const renderStep = () => {
        switch (step) {
            case 0: // Project
                return (
                    <div className="space-y-6">
                        <div>
                            <label className="block text-sm font-medium text-zinc-300 mb-2">Project Name</label>
                            <input
                                type="text"
                                value={projectName}
                                onChange={(e) => setProjectName(e.target.value)}
                                placeholder="e.g. Fantasy Characters"
                                className="w-full px-4 py-3 bg-zinc-900 border border-white/10 rounded-lg text-white placeholder-zinc-500 focus:outline-none focus:border-cyan-500/50"
                                autoFocus
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-zinc-300 mb-2">LoRA Preset</label>
                            <div className="flex gap-3">
                                {['SDXL', 'Flux', 'Face-Only'].map(preset => (
                                    <button
                                        key={preset}
                                        onClick={() => setLoraPreset(preset)}
                                        className={cn(
                                            "px-4 py-2 rounded-lg border transition-all text-sm font-medium",
                                            loraPreset === preset
                                                ? "border-cyan-500 bg-cyan-500/10 text-cyan-400"
                                                : "border-white/10 text-zinc-400 hover:text-white hover:border-white/20"
                                        )}
                                    >
                                        {preset}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>
                );

            case 1: // Character
                return (
                    <div className="space-y-6">
                        <div>
                            <label className="block text-sm font-medium text-zinc-300 mb-2">Character Name</label>
                            <input
                                type="text"
                                value={characterName}
                                onChange={(e) => setCharacterName(e.target.value)}
                                placeholder="e.g. Elara the Mage"
                                className="w-full px-4 py-3 bg-zinc-900 border border-white/10 rounded-lg text-white placeholder-zinc-500 focus:outline-none focus:border-cyan-500/50"
                                autoFocus
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-zinc-300 mb-2">Gender (for body analysis)</label>
                            <div className="flex gap-3">
                                {['neutral', 'female', 'male'].map(g => (
                                    <button
                                        key={g}
                                        onClick={() => setGender(g)}
                                        className={cn(
                                            "px-4 py-2 rounded-lg border transition-all text-sm font-medium capitalize",
                                            gender === g
                                                ? "border-fuchsia-500 bg-fuchsia-500/10 text-fuchsia-400"
                                                : "border-white/10 text-zinc-400 hover:text-white hover:border-white/20"
                                        )}
                                    >
                                        {g}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>
                );

            case 2: // References - New picker UI
                return (
                    <div className="space-y-6">
                        {/* Folder Scanner */}
                        <div className="bg-zinc-900 border border-white/5 rounded-lg p-3">
                            <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">Source Folder</label>
                            <div className="flex gap-2">
                                <input
                                    type="text"
                                    value={refFolderPath}
                                    onChange={(e) => setRefFolderPath(e.target.value)}
                                    placeholder="/path/to/reference/images"
                                    className="flex-1 px-3 py-2 bg-zinc-950 border border-white/10 rounded-lg text-white placeholder-zinc-500 focus:outline-none focus:border-cyan-500/50 font-mono text-sm"
                                />
                                <Button
                                    variant="secondary"
                                    onClick={handleScanRefFolder}
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
                                <label className="block text-xs font-medium text-zinc-400 mb-2 flex justify-between">
                                    <span>Select an image, then click a slot to assign</span>
                                    <span className="text-zinc-600">{availableImages.length} found</span>
                                </label>
                                <div className="bg-zinc-900 rounded-lg border border-white/5 p-3 h-36 overflow-y-hidden overflow-x-auto">
                                    <div className="flex gap-2 h-full">
                                        {availableImages.map((imgPath, i) => {
                                            const isSelected = selectedImage === imgPath;
                                            const isAssigned = Object.values(referenceImages).some(r => r.path === imgPath);
                                            return (
                                                <button
                                                    key={i}
                                                    onClick={() => setSelectedImage(isSelected ? null : imgPath)}
                                                    className={cn(
                                                        "flex-shrink-0 aspect-square h-full rounded-md overflow-hidden border-2 transition-all relative group",
                                                        isSelected
                                                            ? "border-cyan-500 ring-2 ring-cyan-500/30 z-10"
                                                            : isAssigned
                                                                ? "border-green-500/50 opacity-50"
                                                                : "border-white/10 hover:border-white/30"
                                                    )}
                                                >
                                                    <img
                                                        src={api.getThumbnailUrl(imgPath, 128)}
                                                        alt={`Image ${i + 1}`}
                                                        className="w-full h-full object-cover"
                                                    />
                                                    {isSelected && (
                                                        <div className="absolute inset-0 bg-cyan-500/20 flex items-center justify-center">
                                                            <Check className="w-6 h-6 text-cyan-400 drop-shadow-md" />
                                                        </div>
                                                    )}
                                                    {isAssigned && !isSelected && (
                                                        <div className="absolute inset-0 bg-green-500/20 flex items-center justify-center">
                                                            <Check className="w-4 h-4 text-green-400" />
                                                        </div>
                                                    )}
                                                </button>
                                            );
                                        })}
                                    </div>
                                </div>
                            </div>
                        )}

                        <div className="space-y-6">
                            {/* Head References Grid */}
                            <div>
                                <h3 className="text-sm font-medium text-zinc-300 mb-3 flex items-center gap-2">
                                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-500" />
                                    Head References
                                </h3>
                                {renderHeadGrid()}
                            </div>

                            {/* Body References Grid */}
                            <div>
                                <h3 className="text-sm font-medium text-zinc-300 mb-3 flex items-center gap-2">
                                    <span className="w-1.5 h-1.5 rounded-full bg-fuchsia-500" />
                                    Body References
                                </h3>
                                <div className="grid grid-cols-3 gap-3 max-w-lg mx-auto">
                                    {BODY_SLOTS.map(slot => (
                                        <div key={slot.key} className="aspect-[3/4]">
                                            {renderSlotButton(slot)}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                );

            case 3: // Dataset
                return (
                    <div className="space-y-6">
                        <div>
                            <label className="block text-sm font-medium text-zinc-300 mb-2">Dataset Images Folder</label>
                            <div className="flex gap-2">
                                <input
                                    type="text"
                                    value={folderPath}
                                    onChange={(e) => setFolderPath(e.target.value)}
                                    placeholder="/path/to/dataset"
                                    className="flex-1 px-4 py-3 bg-zinc-900 border border-white/10 rounded-lg text-white placeholder-zinc-500 focus:outline-none focus:border-cyan-500/50 font-mono"
                                />
                            </div>
                            <p className="mt-2 text-xs text-zinc-500">
                                Folder containing the images you want to validate against your references.
                            </p>
                        </div>

                        {scanResult && (
                            <div className="bg-zinc-900 rounded-lg p-4 border border-white/5 space-y-3">
                                <div className="flex items-center justify-between">
                                    <span className="text-sm text-zinc-400">Total Images Found</span>
                                    <span className="text-lg font-mono text-white">{scanResult.total_found}</span>
                                </div>
                                <div className="flex items-center justify-between">
                                    <span className="text-sm text-zinc-400">New Entries</span>
                                    <span className="text-lg font-mono text-green-400">+{scanResult.new_entries}</span>
                                </div>
                                <div className="flex items-center justify-between">
                                    <span className="text-sm text-zinc-400">Already in Project</span>
                                    <span className="text-lg font-mono text-zinc-500">{scanResult.already_exists}</span>
                                </div>
                            </div>
                        )}
                    </div>
                );

            case 4: // Validate
                return (
                    <div className="space-y-6 h-full flex flex-col">
                        <div className="text-center space-y-2">
                            <h3 className="text-lg font-medium text-white">Setup Complete</h3>
                            <p className="text-sm text-zinc-400">
                                Your project and character data is ready. Run a final validation check on your references before processing the dataset.
                            </p>
                        </div>

                        <div className="flex-1 bg-zinc-900 rounded-lg border border-white/5 p-6 overflow-y-auto">
                            {!validationResult ? (
                                <div className="h-full flex flex-col items-center justify-center text-center space-y-4">
                                    <div className="w-12 h-12 rounded-full bg-zinc-800 flex items-center justify-center">
                                        <User className="w-6 h-6 text-zinc-500" />
                                    </div>
                                    <p className="text-sm text-zinc-500 max-w-xs">
                                        Validate that your reference images cover all required angles and have consistent identity.
                                    </p>
                                    <Button onClick={handleRunValidation} variant="primary">
                                        Run Reference Validation
                                    </Button>
                                </div>
                            ) : (
                                <div className="space-y-4">
                                    <div className="flex items-center gap-3">
                                        {validationResult.success ? (
                                            <div className="w-8 h-8 rounded-full bg-green-500/20 flex items-center justify-center">
                                                <Check className="w-5 h-5 text-green-500" />
                                            </div>
                                        ) : (
                                            <div className="w-8 h-8 rounded-full bg-yellow-500/20 flex items-center justify-center">
                                                <AlertTriangle className="w-5 h-5 text-yellow-500" />
                                            </div>
                                        )}
                                        <div>
                                            <h4 className={cn("font-medium", validationResult.success ? "text-green-400" : "text-yellow-400")}>
                                                {validationResult.success ? "References Validated" : "Validation Issues"}
                                            </h4>
                                            <p className="text-xs text-zinc-500">
                                                {validationResult.success ? "All consistency checks passed." : "Some issues were detected but you can proceed."}
                                            </p>
                                        </div>
                                    </div>

                                    {validationResult.warnings.length > 0 && (
                                        <div className="space-y-2">
                                            {validationResult.warnings.map((w, i) => (
                                                <div key={i} className="text-xs bg-black/30 p-3 rounded border border-white/5 text-zinc-300">
                                                    <span className={cn("font-bold mr-2 uppercase", w.severity === 'error' ? "text-red-400" : "text-yellow-400")}>
                                                        {w.severity}
                                                    </span>
                                                    {w.message}
                                                </div>
                                            ))}
                                        </div>
                                    )}

                                    <div className="pt-4 flex justify-center">
                                        <Button onClick={handleRunValidation} variant="secondary" size="sm">
                                            Re-run Validation
                                        </Button>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                );
        }
    };

    return (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
            <div className="bg-zinc-950 rounded-xl border border-white/10 w-full max-w-2xl overflow-hidden flex flex-col shadow-2xl max-h-[90vh]">
                {/* Header */}
                <div className="p-6 border-b border-white/5 flex items-center justify-between shrink-0 bg-zinc-900/50">
                    <div>
                        <h2 className="text-xl font-bold text-white tracking-tight">Create New Project</h2>
                        <div className="flex items-center gap-2 mt-2">
                            {STEPS.map((s, i) => (
                                <div key={s} className="flex items-center">
                                    <div className={cn(
                                        "text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full transition-colors",
                                        i === step ? "bg-cyan-500/20 text-cyan-400" :
                                            i < step ? "bg-green-500/20 text-green-400" : "bg-zinc-800 text-zinc-600"
                                    )}>
                                        {i + 1}. {s}
                                    </div>
                                    {i < STEPS.length - 1 && (
                                        <div className="w-4 h-[1px] bg-zinc-800 mx-1" />
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="text-zinc-500 hover:text-white transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Content */}
                <div className="p-6 overflow-y-auto flex-1 bg-zinc-950">
                    {error && (
                        <div className="mb-6 bg-red-500/10 border border-red-500/50 rounded-lg p-4 flex items-start gap-3">
                            <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                            <p className="text-sm text-red-200">{error}</p>
                        </div>
                    )}

                    {renderStep()}
                </div>

                {/* Footer */}
                <div className="p-6 border-t border-white/5 bg-zinc-900 shrink-0 flex justify-between">
                    <Button
                        variant="secondary"
                        onClick={handleBack}
                        disabled={step === (initialProject ? 1 : 0) || isLoading}
                    >
                        <ChevronLeft className="w-4 h-4 mr-1" />
                        Back
                    </Button>
                    <Button
                        variant="primary"
                        onClick={handleNext}
                        disabled={isLoading}
                    >
                        {isLoading && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                        {step === STEPS.length - 1 ? 'Finish' : 'Next'}
                        {!isLoading && step < STEPS.length - 1 && <ChevronRight className="w-4 h-4 ml-1" />}
                    </Button>
                </div>
            </div>
        </div>
    );
}
