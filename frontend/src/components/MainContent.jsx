import { useState } from 'react';
import { Filter, Grid3X3, LayoutList, Image as ImageIcon, Play, X, Loader2, FolderOpen, PlayCircle, Pause, Check, ChevronDown, Pencil } from 'lucide-react';
import { cn } from '../lib/utils';
import { ImageCard } from './ImageCard';
import { Skeleton } from './ui/Skeleton';
import { Button } from './ui/Button';
import { useProject } from '../context/ProjectContext';
import { ImageDetailModal } from './ImageDetailModal';
import EditReferencesModal from './EditReferencesModal';

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
        cancelProcessing
    } = useProject();

    const [selectedImage, setSelectedImage] = useState(null);
    const [statusFilter, setStatusFilter] = useState('all');
    const [reprocessMode, setReprocessMode] = useState(false);
    const [editReferencesOpen, setEditReferencesOpen] = useState(false);

    // Filter images by status
    const filteredImages = statusFilter === 'all'
        ? datasetImages
        : datasetImages.filter(img => img.status === statusFilter);

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

                            {/* Filter Dropdown */}
                            <div className="flex items-center gap-1 text-sm">
                                <Filter className="w-3.5 h-3.5 text-zinc-500" />
                                <select
                                    value={statusFilter}
                                    onChange={(e) => setStatusFilter(e.target.value)}
                                    className="bg-transparent text-zinc-400 border-none focus:outline-none cursor-pointer"
                                >
                                    <option value="all">All ({statusCounts.all})</option>
                                    <option value="pending">Pending ({statusCounts.pending})</option>
                                    <option value="approved">Approved ({statusCounts.approved})</option>
                                    <option value="analyzed">Analyzed ({statusCounts.analyzed})</option>
                                    <option value="rejected">Rejected ({statusCounts.rejected})</option>
                                </select>
                            </div>

                            {/* View Toggle */}
                            <div className="flex rounded-md overflow-hidden border border-white/10 p-0.5 bg-zinc-900/50">
                                <button className="p-1.5 rounded bg-white/10 text-cyan-400">
                                    <Grid3X3 className="w-3.5 h-3.5" />
                                </button>
                                <button className="p-1.5 rounded hover:bg-white/5 text-zinc-500 hover:text-zinc-300 transition-colors">
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
                        ) : (
                            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6 p-2">
                                {filteredImages.map((image) => (
                                    <ImageCard
                                        key={image.id}
                                        image={image}
                                        isSelected={selectedImage?.id === image.id}
                                        onClick={() => setSelectedImage(image)}
                                    />
                                ))}
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
                        face_bbox: selectedImage.face_bbox
                    }}
                />
            )}

            {/* Edit References Modal */}
            {selectedCharacter && (
                <EditReferencesModal
                    character={selectedCharacter}
                    open={editReferencesOpen}
                    onClose={() => setEditReferencesOpen(false)}
                    onSave={() => {
                        // Refresh character data
                        if (selectedCharacter) {
                            // The context will auto-refresh on next render
                            setEditReferencesOpen(false);
                        }
                    }}
                />
            )}
        </>
    );
}
