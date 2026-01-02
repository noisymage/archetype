import { useState, useRef } from 'react';
import { X, FolderOpen, User, ChevronRight, ChevronLeft, Check, Upload, Loader2, AlertTriangle, Search } from 'lucide-react';
import { cn } from '../lib/utils';
import { Button } from './ui/Button';
import { useProject } from '../context/ProjectContext';
import * as api from '../lib/api';

/**
 * Reference image slot configurations
 */
const headSlots = [
    { key: 'head_front', label: 'Front', icon: '⚫', description: 'Face camera', required: true },
    { key: 'head_45l', label: '45° Left', icon: '←', description: "Viewer's left", required: true },
    { key: 'head_45r', label: '45° Right', icon: '→', description: "Viewer's right", required: true },
    { key: 'head_90l', label: '90° Left Profile', icon: '⮜', description: 'Full left side', required: false, optional: true },
    { key: 'head_90r', label: '90° Right Profile', icon: '⮞', description: 'Full right side', required: false, optional: true },
    { key: 'head_up', label: 'Looking Up', icon: '⮝', description: 'Pitch +30°', required: false, optional: true },
    { key: 'head_down', label: 'Looking Down', icon: '⮟', description: 'Pitch -30°', required: false, optional: true },
];

const BODY_SLOTS = [
    { key: 'body_front', label: 'A-Pose Front', icon: '╋' },
    { key: 'body_side', label: 'Side Profile', icon: '│' },
    { key: 'body_posterior', label: 'Posterior', icon: '◎', description: 'Back view' }
];

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
            setAvailableImages(result.images || []);
            if (result.images?.length === 0) {
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
                    const requiredSlots = [...headSlots.filter(s => s.required), ...BODY_SLOTS];
                    const filledRequired = requiredSlots.filter(s => referenceImages[s.key]);
                    if (filledRequired.length < 6) {
                        throw new Error(`Please assign all 6 required reference images (${filledRequired.length}/6 filled)`);
                    }
                    // Save reference paths
                    const allSlots = [...headSlots, ...BODY_SLOTS]; // Keep allSlots for saving all assigned images
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
                        for (const slot of [...headSlots, ...BODY_SLOTS]) {
                            paths[slot.key] = referenceImages[slot.key].path;
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
            for (const slot of [...headSlots, ...BODY_SLOTS]) {
                paths[slot.key] = referenceImages[slot.key].path;
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

        return (
            <button
                key={slot.key}
                onClick={() => handleAssignToSlot(slot.key)}
                disabled={!selectedImage && !assigned}
                className={cn(
                    "flex flex-col items-center p-2 rounded-lg border-2 transition-all",
                    assigned
                        ? "border-green-500/50 bg-green-500/10"
                        : isTarget
                            ? "border-cyan-500 bg-cyan-500/20 animate-pulse cursor-pointer"
                            : "border-white/10 bg-white/5",
                    isTarget && "hover:border-cyan-400"
                )}
            >
                <div className="w-16 h-16 rounded overflow-hidden bg-zinc-800 flex items-center justify-center">
                    {assigned ? (
                        <img
                            src={api.getThumbnailUrl(assigned.path, 64)}
                            alt={slot.label}
                            className="w-full h-full object-cover"
                        />
                    ) : (
                        <span className="text-xl text-zinc-600">{slot.icon}</span>
                    )}
                </div>
                <span className="mt-1 text-[10px] text-zinc-400 font-medium">{slot.label}</span>
                {assigned && (
                    <button
                        onClick={(e) => {
                            e.stopPropagation();
                            setReferenceImagesState(prev => {
                                const next = { ...prev };
                                delete next[slot.key];
                                return next;
                            });
                        }}
                        className="mt-1 text-[9px] text-red-400 hover:text-red-300"
                    >
                        Clear
                    </button>
                )}
            </button>
        );
    };

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
                        {/* Head References */}
                        <div>
                            <h3 className="text-sm font-medium text-zinc-300 mb-3">Head References</h3>
                            <div className="grid grid-cols-3 gap-3">
                                {headSlots.filter(s => s.required).map(slot => (
                                    <ReferenceSlot
                                        key={slot.key}
                                        slot={slot}
                                        selectedImages={selectedImages}
                                        onSelect={() => handleSelectReference(slot.key)}
                                    />
                                ))}
                            </div>

                            {/* Optional head references */}
                            <div className="mt-4">
                                <h4 className="text-xs font-medium text-zinc-500 mb-2">Optional (Improves accuracy)</h4>
                                <div className="grid grid-cols-4 gap-2">
                                    {headSlots.filter(s => s.optional).map(slot => (
                                        <ReferenceSlot
                                            key={slot.key}
                                            slot={slot}
                                            selectedImages={selectedImages}
                                            onSelect={() => handleSelectReference(slot.key)}
                                            optional={true}
                                        />
                                    ))}
                                </div>
                            </div>
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
                    <div className="space-y-4">
                        {/* Folder Scanner */}
                        <div>
                            <label className="block text-sm font-medium text-zinc-300 mb-2">Reference Images Folder</label>
                            <div className="flex gap-2">
                                <input
                                    type="text"
                                    value={refFolderPath}
                                    onChange={(e) => setRefFolderPath(e.target.value)}
                                    placeholder="/path/to/reference/images"
                                    className="flex-1 px-3 py-2 bg-zinc-900 border border-white/10 rounded-lg text-white placeholder-zinc-500 focus:outline-none focus:border-cyan-500/50 font-mono text-sm"
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
                                <label className="block text-xs font-medium text-zinc-500 mb-2">
                                    Select an image, then click a slot to assign ({availableImages.length} found)
                                </label>
                                <div className="h-32 overflow-x-auto overflow-y-hidden">
                                    <div className="flex gap-2 pb-2">
                                        {availableImages.map((imgPath, i) => {
                                            const isSelected = selectedImage === imgPath;
                                            const isAssigned = Object.values(referenceImages).some(r => r.path === imgPath);
                                            return (
                                                <button
                                                    key={i}
                                                    onClick={() => setSelectedImage(isSelected ? null : imgPath)}
                                                    className={cn(
                                                        "flex-shrink-0 w-24 h-24 rounded-lg overflow-hidden border-2 transition-all",
                                                        isSelected
                                                            ? "border-cyan-500 ring-2 ring-cyan-500/50"
                                                            : isAssigned
                                                                ? "border-green-500/50 opacity-50"
                                                                : "border-white/10 hover:border-white/30"
                                                    )}
                                                >
                                                    <img
                                                        src={api.getThumbnailUrl(imgPath, 96)}
                                                        alt={`Image ${i + 1}`}
                                                        className="w-full h-full object-cover"
                                                    />
                                                </button>
                                            );
                                        })}
                                    </div>
                                </div>
                                {selectedImage && (
                                    <p className="text-xs text-cyan-400 mt-1">
                                        ✓ Image selected. Click a slot below to assign it.
                                    </p>
                                )}
                            </div>
                        )}

                        {/* Slot Assignment */}
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <h4 className="text-xs font-medium text-zinc-400 mb-2 flex items-center gap-2">
                                    <User className="w-3 h-3" />
                                    Head References
                                </h4>
                                <div className="flex gap-2">
                                    {HEAD_SLOTS.map(renderSlotButton)}
                                </div>
                            </div>
                            <div>
                                <h4 className="text-xs font-medium text-zinc-400 mb-2 flex items-center gap-2">
                                    <span>🧍</span>
                                    Body References
                                </h4>
                                <div className="flex gap-2">
                                    {BODY_SLOTS.map(renderSlotButton)}
                                </div>
                            </div>
                        </div>

                        <p className="text-[10px] text-zinc-600 text-center">
                            {Object.keys(referenceImages).length}/6 slots assigned
                        </p>
                    </div>
                );

            case 3: // Dataset
                return (
                    <div className="space-y-6">
                        <div>
                            <label className="block text-sm font-medium text-zinc-300 mb-2">Dataset Folder Path</label>
                            <div className="flex gap-2">
                                <input
                                    type="text"
                                    value={folderPath}
                                    onChange={(e) => setFolderPath(e.target.value)}
                                    placeholder="/path/to/dataset/images"
                                    className="flex-1 px-4 py-3 bg-zinc-900 border border-white/10 rounded-lg text-white placeholder-zinc-500 focus:outline-none focus:border-cyan-500/50 font-mono text-sm"
                                />
                                <Button
                                    variant="secondary"
                                    className="gap-2"
                                    onClick={async () => {
                                        if (!folderPath.trim()) return;
                                        setIsLoading(true);
                                        setError(null);
                                        try {
                                            const result = await scanFolder(folderPath.trim(), createdCharacter.id);
                                            setScanResult(result);
                                        } catch (err) {
                                            setError(err.message);
                                        } finally {
                                            setIsLoading(false);
                                        }
                                    }}
                                    disabled={isLoading || !folderPath.trim()}
                                >
                                    {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                                    Scan
                                </Button>
                            </div>
                            <p className="text-xs text-zinc-500 mt-2">
                                Enter the absolute path to the folder containing training images.
                            </p>
                        </div>

                        {scanResult && (
                            <div className="p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
                                <div className="flex items-center gap-2 text-green-400 mb-2">
                                    <Check className="w-4 h-4" />
                                    <span className="font-medium">Folder Scanned</span>
                                </div>
                                <div className="text-sm text-zinc-300 space-y-1">
                                    <p>Found: <span className="text-white font-mono">{scanResult.total_found}</span> images</p>
                                    <p>New entries: <span className="text-white font-mono">{scanResult.new_entries}</span></p>
                                    {scanResult.already_exists > 0 && (
                                        <p className="text-zinc-500">({scanResult.already_exists} already imported)</p>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                );

            case 4: // Validate
                return (
                    <div className="space-y-6">
                        <div className="text-center">
                            <h4 className="text-lg font-medium text-white mb-2">Reference Validation</h4>
                            <p className="text-sm text-zinc-400">
                                Analyze reference images for identity consistency and body proportions.
                            </p>
                        </div>

                        {!validationResult && !isLoading && (
                            <div className="flex justify-center">
                                <Button variant="primary" onClick={handleRunValidation} className="gap-2">
                                    <Check className="w-4 h-4" />
                                    Run Validation
                                </Button>
                            </div>
                        )}

                        {isLoading && (
                            <div className="flex flex-col items-center gap-3 py-8">
                                <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
                                <p className="text-sm text-zinc-400">Analyzing references...</p>
                            </div>
                        )}

                        {validationResult && (
                            <div className={cn(
                                "p-4 rounded-lg border",
                                validationResult.success
                                    ? "bg-green-500/10 border-green-500/30"
                                    : "bg-yellow-500/10 border-yellow-500/30"
                            )}>
                                <div className="flex items-center gap-2 mb-3">
                                    {validationResult.success ? (
                                        <Check className="w-5 h-5 text-green-400" />
                                    ) : (
                                        <AlertTriangle className="w-5 h-5 text-yellow-400" />
                                    )}
                                    <span className={cn(
                                        "font-medium",
                                        validationResult.success ? "text-green-400" : "text-yellow-400"
                                    )}>
                                        {validationResult.success ? 'Validation Passed' : 'Validation Warnings'}
                                    </span>
                                    {validationResult.degraded_mode && (
                                        <span className="text-xs px-2 py-0.5 bg-yellow-500/20 text-yellow-400 rounded">
                                            Degraded Mode
                                        </span>
                                    )}
                                </div>

                                {validationResult.warnings?.length > 0 && (
                                    <div className="space-y-2">
                                        {validationResult.warnings.map((warning, i) => (
                                            <div key={i} className={cn(
                                                "text-sm px-3 py-2 rounded",
                                                warning.severity === 'error'
                                                    ? "bg-red-500/10 text-red-400"
                                                    : "bg-yellow-500/10 text-yellow-400"
                                            )}>
                                                <span className="font-mono text-xs opacity-50">[{warning.code}]</span>{' '}
                                                {warning.message}
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                );

            default:
                return null;
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
            {/* Backdrop */}
            <div className="absolute inset-0 bg-black/80 backdrop-blur-sm" onClick={onClose} />

            {/* Modal */}
            <div className="relative w-full max-w-2xl bg-zinc-950 border border-white/10 rounded-2xl shadow-2xl overflow-hidden">
                {/* Header */}
                <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between">
                    <h2 className="text-lg font-semibold text-white">
                        {initialProject ? `Add Character to ${initialProject.name}` : 'Create New Project'}
                    </h2>
                    <button
                        onClick={onClose}
                        className="p-2 rounded-lg hover:bg-white/5 text-zinc-400 hover:text-white transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Step Indicator */}
                <div className="px-6 py-4 border-b border-white/5">
                    <div className="flex items-center justify-between">
                        {STEPS.map((name, i) => (
                            <div key={i} className="flex items-center">
                                <div className={cn(
                                    "w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-all",
                                    i < step ? "bg-cyan-500 text-black" :
                                        i === step ? "bg-cyan-500/20 text-cyan-400 ring-2 ring-cyan-500" :
                                            "bg-zinc-800 text-zinc-500"
                                )}>
                                    {i < step ? <Check className="w-4 h-4" /> : i + 1}
                                </div>
                                <span className={cn(
                                    "ml-2 text-sm hidden sm:block",
                                    i === step ? "text-white" : "text-zinc-500"
                                )}>
                                    {name}
                                </span>
                                {i < STEPS.length - 1 && (
                                    <ChevronRight className="w-4 h-4 text-zinc-600 mx-3" />
                                )}
                            </div>
                        ))}
                    </div>
                </div>

                {/* Content */}
                <div className="px-6 py-8 min-h-[300px]">
                    {renderStep()}

                    {error && (
                        <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
                            {error}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="px-6 py-4 border-t border-white/10 flex items-center justify-between">
                    <Button
                        variant="secondary"
                        onClick={handleBack}
                        disabled={step === 0 || isLoading}
                        className="gap-2"
                    >
                        <ChevronLeft className="w-4 h-4" />
                        Back
                    </Button>

                    <Button
                        variant="primary"
                        onClick={handleNext}
                        disabled={isLoading}
                        className="gap-2"
                    >
                        {isLoading ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                        ) : step === STEPS.length - 1 ? (
                            <>
                                <Check className="w-4 h-4" />
                                Complete
                            </>
                        ) : (
                            <>
                                Next
                                <ChevronRight className="w-4 h-4" />
                            </>
                        )}
                    </Button>
                </div>
            </div>
        </div>
    );
}
