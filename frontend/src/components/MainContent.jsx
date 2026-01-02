import { useState, useEffect, useMemo } from 'react';
import { Filter, Grid3X3, LayoutList, Image as ImageIcon, Play, X, Loader2, FolderOpen, PlayCircle, Pause, Check, ChevronDown, Pencil } from 'lucide-react';
import { cn } from '../lib/utils';
import { ImageCard } from './ImageCard';
import { Skeleton } from './ui/Skeleton';
import { Button } from './ui/Button';
import { useProject } from '../context/ProjectContext';
import { ImageDetailModal } from './ImageDetailModal';
import EditReferencesModal from './EditReferencesModal';
import { ScanFolderModal } from './ScanFolderModal';
import * as api from '../lib/api';

/**
 * Main content area with responsive image grid
 */
export function MainContent() {
    const {
        selectedCharacter,
        datasetImages,
        isLoading,
        activeJob,
        startProcessing,
        cancelProcessing,
        loadDatasetImages
    } = useProject();

    const [selectedImage, setSelectedImage] = useState(null);
    const [statusFilter, setStatusFilter] = useState('all');
    const [sortBy, setSortBy] = useState('date');
    const [viewMode, setViewMode] = useState('grid');
    const [reprocessMode, setReprocessMode] = useState(false);
    const [editReferencesOpen, setEditReferencesOpen] = useState(false);
    const [scanModalOpen, setScanModalOpen] = useState(false);

    // Local state for references (fetched separately)
    const [characterReferences, setCharacterReferences] = useState([]);

    // Fetch references when character changes
    useEffect(() => {
        if (selectedCharacter) {
            api.getCharacterReferences(selectedCharacter.id)
                .then(refs => setCharacterReferences(refs))
                .catch(err => console.error("Failed to load references:", err));
        } else {
            setCharacterReferences([]);
        }
    }, [selectedCharacter?.id]);

    // Filter images by status
    const filteredImages = statusFilter === 'all'
        ? datasetImages
        : datasetImages.filter(img => img.status === statusFilter);

    // Sort images
    const sortedImages = useMemo(() => {
        return [...filteredImages].sort((a, b) => {
            switch (sortBy) {
                case 'date':
                    return (b.id || 0) - (a.id || 0); // Assuming ID is roughly chronological
                case 'date_asc':
                    return (a.id || 0) - (b.id || 0);
                case 'face_sim':
                    return (b.face_similarity || 0) - (a.face_similarity || 0);
                case 'body_sim':
                    return (b.body_consistency || 0) - (a.body_consistency || 0);
                case 'status':
                    return a.status.localeCompare(b.status);
                default:
                    return 0;
            }
        });
    }, [filteredImages, sortBy]);

    // Count by status
    const statusCounts = {
        all: datasetImages.length,
        pending: datasetImages.filter(i => i.status === 'pending').length,
        approved: datasetImages.filter(i => i.status === 'approved').length,
        analyzed: datasetImages.filter(i => i.status === 'analyzed').length,
        rejected: datasetImages.filter(i => i.status === 'rejected').length
    };

    const handleStartProcessing = async () => {
        if (selectedCharacter) {
            try {
                await startProcessing(selectedCharacter.id, reprocessMode);
            } catch (error) {
                // Error already shown via toast
            }
        }
    };

    // Map references to simplified { view_type: path } format for the modal
    const referencesMap = useMemo(() => {
        const map = {};
        if (Array.isArray(characterReferences)) {
            characterReferences.forEach(ref => {
                map[ref.view_type] = ref.path;
            });
        }
        return map;
    }, [characterReferences]);

    return (
        <>
            <main className="flex-1 flex flex-col bg-zinc-950 overflow-hidden relative">
                {/* Background Grid Pattern */}
                <div className="absolute inset-0 bg-[url('/grid-pattern.svg')] opacity-[0.02] pointer-events-none" />

                {/* Toolbar */}
                <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between z-10 bg-zinc-950/50 backdrop-blur-sm">
                    <div>
                        <h2 className="text-xl font-semibold text-zinc-100 tracking-tight">
                            {selectedCharacter?.name || 'Select a Character'}
                        </h2>
                        {selectedCharacter ? (
                            <p className="text-xs text-zinc-500 mt-1 font-mono">
                                ID: <span className="text-zinc-400">{selectedCharacter.id.toString().padStart(4, '0')}</span> • {datasetImages.length} IMAGES
                            </p>
                        ) : (
                            <Skeleton className="h-4 w-32 mt-1 bg-zinc-800/50" />
                        )}
                    </div>

                    {selectedCharacter && (
                        <div className="flex items-center gap-3">
                            {/* Reprocess Toggle */}
                            {datasetImages.length > 0 && !activeJob && (
                                <label className="flex items-center gap-2 cursor-pointer mr-2 select-none group">
                                    <div className="relative flex items-center">
                                        <input
                                            type="checkbox"
                                            checked={reprocessMode}
                                            onChange={(e) => setReprocessMode(e.target.checked)}
                                            className="peer appearance-none w-4 h-4 rounded border border-zinc-700 bg-zinc-900 checked:bg-cyan-500 checked:border-cyan-500 transition-colors focus:ring-2 focus:ring-cyan-500/20 focus:outline-none"
                                        />
                                        <svg className="absolute w-2.5 h-2.5 text-black pointer-events-none opacity-0 peer-checked:opacity-100 left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 transition-opacity" viewBox="0 0 12 12" fill="none">
                                            <path d="M10 3L4.5 8.5L2 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                        </svg>
                                    </div>
                                    <span className="text-xs text-zinc-500 group-hover:text-zinc-300 transition-colors">Reprocess All</span>
                                </label>
                            )}

                            {/* Process Button */}
                            {(statusCounts.pending > 0 || reprocessMode) && !activeJob && (
                                <Button
                                    variant="primary"
                                    size="sm"
                                    className="gap-2"
                                    onClick={handleStartProcessing}
                                >
                                    <Play className="w-3.5 h-3.5" />
                                    Process ({reprocessMode ? datasetImages.length : statusCounts.pending})
                                </Button>
                            )}

                            {/* Scan Button */}
                            {!activeJob && (
                                <Button
                                    variant={datasetImages.length === 0 ? "primary" : "secondary"}
                                    size="sm"
                                    className="gap-2"
                                    onClick={() => setScanModalOpen(true)}
                                >
                                    <FolderOpen className="w-3.5 h-3.5" />
                                    {datasetImages.length === 0 ? "Scan Image Folder" : "Rescan"}
                                </Button>
                            )}

                            {/* Filter Dropdown */}
                            <div className="flex items-center gap-1 text-sm bg-zinc-900/50 rounded-md border border-white/10 px-2 py-1">
                                <Filter className="w-3.5 h-3.5 text-zinc-500" />
                                <select
                                    value={statusFilter}
                                    onChange={(e) => setStatusFilter(e.target.value)}
                                    className="bg-transparent text-zinc-400 border-none focus:outline-none cursor-pointer text-xs"
                                >
                                    <option value="all">All ({statusCounts.all})</option>
                                    <option value="pending">Pending ({statusCounts.pending})</option>
                                    <option value="approved">Approved ({statusCounts.approved})</option>
                                    <option value="analyzed">Analyzed ({statusCounts.analyzed})</option>
                                    <option value="rejected">Rejected ({statusCounts.rejected})</option>
                                </select>
                            </div>

                            {/* Sort Dropdown */}
                            <div className="flex items-center gap-1 text-sm bg-zinc-900/50 rounded-md border border-white/10 px-2 py-1">
                                <ChevronDown className="w-3.5 h-3.5 text-zinc-500" />
                                <select
                                    value={sortBy}
                                    onChange={(e) => setSortBy(e.target.value)}
                                    className="bg-transparent text-zinc-400 border-none focus:outline-none cursor-pointer text-xs"
                                >
                                    <option value="date">Newest First</option>
                                    <option value="date_asc">Oldest First</option>
                                    <option value="face_sim">Face Match</option>
                                    <option value="body_sim">Body Match</option>
                                    <option value="status">Status</option>
                                </select>
                            </div>

                            {/* View Toggle */}
                            <div className="flex rounded-md overflow-hidden border border-white/10 p-0.5 bg-zinc-900/50">
                                <button
                                    onClick={() => setViewMode('grid')}
                                    className={cn(
                                        "p-1.5 rounded transition-colors",
                                        viewMode === 'grid' ? "bg-white/10 text-cyan-400" : "hover:bg-white/5 text-zinc-500 hover:text-zinc-300"
                                    )}
                                >
                                    <Grid3X3 className="w-3.5 h-3.5" />
                                </button>
                                <button
                                    onClick={() => setViewMode('list')}
                                    className={cn(
                                        "p-1.5 rounded transition-colors",
                                        viewMode === 'list' ? "bg-white/10 text-cyan-400" : "hover:bg-white/5 text-zinc-500 hover:text-zinc-300"
                                    )}
                                >
                                    <LayoutList className="w-3.5 h-3.5" />
                                </button>
                            </div>
                        </div>
                    )}
                </div>

                {/* Progress Bar (when processing) */}
                {activeJob && (
                    <div className="px-6 py-3 border-b border-white/5 bg-zinc-900/50">
                        <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-3">
                                <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />
                                <span className="text-sm text-zinc-300">
                                    Processing {activeJob.processed || 0}/{activeJob.total || activeJob.total_images}
                                </span>
                                {activeJob.currentImage && (
                                    <span className="text-xs text-zinc-500 font-mono">
                                        {activeJob.currentImage}
                                    </span>
                                )}
                            </div>
                            <Button
                                variant="secondary"
                                size="sm"
                                onClick={cancelProcessing}
                                className="gap-1 text-red-400 hover:text-red-300"
                            >
                                <X className="w-3.5 h-3.5" />
                                Cancel
                            </Button>
                        </div>
                        <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-gradient-to-r from-cyan-500 to-fuchsia-500 rounded-full transition-all duration-300"
                                style={{
                                    width: `${activeJob.total ? ((activeJob.processed || 0) / activeJob.total) * 100 : 0}%`
                                }}
                            />
                        </div>
                    </div>
                )}

                {/* Edit References Button */}
                {selectedCharacter && !activeJob && (
                    <div className="px-6 py-3 border-b border-white/5 bg-zinc-900/50 flex justify-end">
                        <Button
                            variant="secondary"
                            size="sm"
                            className="gap-2"
                            onClick={() => setEditReferencesOpen(true)}
                        >
                            <Pencil className="w-3.5 h-3.5" />
                            Edit References
                        </Button>
                    </div>
                )}

                {/* Image Grid */}
                <div className="flex-1 overflow-y-auto p-8 z-10">
                    {selectedCharacter ? (
                        isLoading ? (
                            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
                                {Array.from({ length: 12 }).map((_, i) => (
                                    <Skeleton key={i} className="aspect-square rounded-xl bg-zinc-800/40" />
                                ))}
                            </div>
                        ) : filteredImages.length === 0 ? (
                            <div className="h-full flex flex-col items-center justify-center text-center">
                                <div className="w-24 h-24 rounded-3xl bg-zinc-900/50 border border-white/5 flex items-center justify-center mb-6 shadow-2xl shadow-black/50">
                                    <ImageIcon className="w-10 h-10 text-zinc-700" />
                                </div>
                                <h3 className="text-lg font-medium text-zinc-300 mb-2">
                                    {statusFilter !== 'all' ? 'No images match filter' : 'No Dataset Images'}
                                </h3>
                                <p className="text-sm text-zinc-500 max-w-xs leading-relaxed">
                                    {statusFilter !== 'all'
                                        ? 'Try changing the status filter above.'
                                        : 'Scan a folder to import images for this character.'}
                                </p>
                            </div>
                        ) : viewMode === 'grid' ? (
                            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6 p-2">
                                {sortedImages.map((image) => (
                                    <ImageCard
                                        key={image.id}
                                        image={image}
                                        isSelected={selectedImage?.id === image.id}
                                        onClick={() => setSelectedImage(image)}
                                    />
                                ))}
                            </div>
                        ) : (
                            <div className="flex flex-col min-w-full">
                                <div className="grid grid-cols-[80px_1fr_100px_100px_120px_120px] gap-4 px-4 py-2 border-b border-white/10 text-xs font-medium text-zinc-500 uppercase tracking-wider sticky top-0 bg-zinc-950 z-10">
                                    <div>Image</div>
                                    <div>Filename</div>
                                    <div>Status</div>
                                    <div>Shot Type</div>
                                    <div>Face Match</div>
                                    <div>Body Match</div>
                                </div>
                                <div className="divide-y divide-white/5">
                                    {sortedImages.map((image) => (
                                        <div
                                            key={image.id}
                                            onClick={() => setSelectedImage(image)}
                                            className={cn(
                                                "grid grid-cols-[80px_1fr_100px_100px_120px_120px] gap-4 px-4 py-3 items-center hover:bg-white/5 cursor-pointer transition-colors group",
                                                selectedImage?.id === image.id && "bg-cyan-500/10 hover:bg-cyan-500/15"
                                            )}
                                        >
                                            <div className="relative aspect-square w-12 h-12 rounded overflow-hidden bg-zinc-800">
                                                <img
                                                    src={api.getThumbnailUrl(image.original_path || image.path, 256)}
                                                    alt={image.filename}
                                                    className="w-full h-full object-cover"
                                                />
                                            </div>
                                            <div className="truncate text-sm text-zinc-300 font-mono">
                                                {image.filename}
                                            </div>
                                            <div>
                                                <span className={cn(
                                                    "px-2 py-0.5 rounded text-[10px] font-medium uppercase tracking-wider",
                                                    image.status === 'approved' && "bg-emerald-500/20 text-emerald-400",
                                                    image.status === 'rejected' && "bg-red-500/20 text-red-400",
                                                    image.status === 'analyzed' && "bg-blue-500/20 text-blue-400",
                                                    image.status === 'pending' && "bg-zinc-500/20 text-zinc-400"
                                                )}>
                                                    {image.status}
                                                </span>
                                            </div>
                                            <div className="text-xs text-zinc-400 uppercase">
                                                {image.shot_type || '-'}
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                                                    <div
                                                        className="h-full bg-cyan-500 rounded-full"
                                                        style={{ width: `${(image.face_similarity || 0) * 100}%` }}
                                                    />
                                                </div>
                                                <span className="text-xs text-zinc-500 w-8 text-right">
                                                    {image.face_similarity ? Math.round(image.face_similarity * 100) : 0}%
                                                </span>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                                                    <div
                                                        className="h-full bg-fuchsia-500 rounded-full"
                                                        style={{ width: `${(image.body_consistency || 0) * 100}%` }}
                                                    />
                                                </div>
                                                <span className="text-xs text-zinc-500 w-8 text-right">
                                                    {image.body_consistency ? Math.round(image.body_consistency * 100) : 0}%
                                                </span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )
                    ) : (
                        <div className="h-full flex flex-col items-center justify-center text-center opacity-0 animate-[fadeIn_0.5s_ease-out_forwards]">
                            <div className="w-24 h-24 rounded-3xl bg-zinc-900/50 border border-white/5 flex items-center justify-center mb-6 shadow-2xl shadow-black/50">
                                <ImageIcon className="w-10 h-10 text-zinc-700" />
                            </div>
                            <h3 className="text-lg font-medium text-zinc-300 mb-2">
                                No Character Selected
                            </h3>
                            <p className="text-sm text-zinc-500 max-w-xs leading-relaxed">
                                Select a character from the sidebar to view and manage their dataset images.
                            </p>
                        </div>
                    )}
                </div>
            </main>

            {/* Image Detail Modal */}
            {selectedImage && (
                <ImageDetailModal
                    image={selectedImage}
                    open={!!selectedImage}
                    onClose={() => setSelectedImage(null)}
                    metrics={{
                        keypoints: selectedImage.keypoints,
                        face_bbox: selectedImage.face_bbox,
                        shot_type: selectedImage.shot_type
                    }}
                    references={referencesMap}
                    onUpdate={(updatedImage) => {
                        setSelectedImage(updatedImage);
                        // Refresh grid
                        if (selectedCharacter) loadDatasetImages(selectedCharacter.id);
                    }}
                />
            )}

            {/* Edit References Modal */}
            {selectedCharacter && (
                <EditReferencesModal
                    character={selectedCharacter}
                    open={editReferencesOpen}
                    onClose={() => setEditReferencesOpen(false)}
                    onSave={async (newPaths) => {
                        await setReferenceImages(selectedCharacter.id, newPaths);
                        // Refresh local references after save
                        const updatedRefs = await api.getCharacterReferences(selectedCharacter.id);
                        setCharacterReferences(updatedRefs);
                        setEditReferencesOpen(false);
                    }}
                />
            )}

            {/* Scan Folder Modal */}
            {selectedCharacter && (
                <ScanFolderModal
                    character={selectedCharacter}
                    open={scanModalOpen}
                    onClose={() => setScanModalOpen(false)}
                />
            )}
        </>
    );
}
