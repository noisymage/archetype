import { Image, CheckCircle, XCircle, Clock, Filter, Grid3X3, LayoutList } from 'lucide-react';
import { cn } from '../lib/utils';

// Mock image data
const mockImages = Array.from({ length: 12 }, (_, i) => ({
    id: i + 1,
    status: ['pending', 'analyzed', 'approved', 'rejected'][Math.floor(Math.random() * 4)],
    score: Math.random() * 0.4 + 0.6, // 0.6 - 1.0 range
}));

const statusConfig = {
    pending: { icon: Clock, color: 'text-yellow-500', bg: 'bg-yellow-500/10' },
    analyzed: { icon: CheckCircle, color: 'text-blue-500', bg: 'bg-blue-500/10' },
    approved: { icon: CheckCircle, color: 'text-emerald-500', bg: 'bg-emerald-500/10' },
    rejected: { icon: XCircle, color: 'text-red-500', bg: 'bg-red-500/10' },
};

/**
 * Main content area with responsive image grid
 */
export function MainContent({ selectedCharacter, onSelectImage, selectedImage }) {
    return (
        <main className="flex-1 flex flex-col bg-[var(--color-bg-primary)] overflow-hidden">
            {/* Toolbar */}
            <div className="px-6 py-4 border-b border-[var(--color-border)] flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-semibold text-[var(--color-text-primary)]">
                        {selectedCharacter?.name || 'Select a Character'}
                    </h2>
                    {selectedCharacter && (
                        <p className="text-sm text-[var(--color-text-secondary)] mt-0.5">
                            {selectedCharacter.imageCount} images in dataset
                        </p>
                    )}
                </div>

                {selectedCharacter && (
                    <div className="flex items-center gap-2">
                        <button className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[var(--color-bg-tertiary)] hover:bg-[var(--color-bg-elevated)] border border-[var(--color-border)] transition-colors">
                            <Filter className="w-4 h-4 text-[var(--color-text-secondary)]" />
                            <span className="text-sm text-[var(--color-text-secondary)]">Filter</span>
                        </button>
                        <div className="flex rounded-lg overflow-hidden border border-[var(--color-border)]">
                            <button className="p-2 bg-[var(--color-accent-muted)] text-[var(--color-accent)]">
                                <Grid3X3 className="w-4 h-4" />
                            </button>
                            <button className="p-2 bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-elevated)] transition-colors">
                                <LayoutList className="w-4 h-4" />
                            </button>
                        </div>
                    </div>
                )}
            </div>

            {/* Image Grid */}
            <div className="flex-1 overflow-y-auto p-6">
                {selectedCharacter ? (
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                        {mockImages.map((image) => {
                            const StatusIcon = statusConfig[image.status].icon;
                            return (
                                <button
                                    key={image.id}
                                    onClick={() => onSelectImage(image)}
                                    className={cn(
                                        'group relative aspect-square rounded-xl overflow-hidden border-2 transition-all hover:scale-[1.02] hover:shadow-xl',
                                        selectedImage?.id === image.id
                                            ? 'border-[var(--color-accent)] shadow-lg shadow-indigo-500/20'
                                            : 'border-[var(--color-border)] hover:border-[var(--color-border-hover)]'
                                    )}
                                >
                                    {/* Placeholder image background */}
                                    <div className="absolute inset-0 bg-gradient-to-br from-[var(--color-bg-tertiary)] to-[var(--color-bg-elevated)] flex items-center justify-center">
                                        <Image className="w-12 h-12 text-[var(--color-text-muted)]" />
                                    </div>

                                    {/* Status badge */}
                                    <div
                                        className={cn(
                                            'absolute top-2 right-2 p-1.5 rounded-lg backdrop-blur-sm',
                                            statusConfig[image.status].bg
                                        )}
                                    >
                                        <StatusIcon
                                            className={cn('w-4 h-4', statusConfig[image.status].color)}
                                        />
                                    </div>

                                    {/* Score overlay on hover */}
                                    <div className="absolute inset-x-0 bottom-0 p-3 bg-gradient-to-t from-black/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
                                        <div className="flex items-center justify-between text-xs">
                                            <span className="text-white/80">Score</span>
                                            <span className="text-white font-medium">
                                                {(image.score * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                        <div className="mt-1.5 h-1.5 bg-white/20 rounded-full overflow-hidden">
                                            <div
                                                className="h-full bg-gradient-to-r from-emerald-400 to-emerald-500 rounded-full"
                                                style={{ width: `${image.score * 100}%` }}
                                            />
                                        </div>
                                    </div>
                                </button>
                            );
                        })}
                    </div>
                ) : (
                    <div className="h-full flex flex-col items-center justify-center text-center">
                        <div className="w-20 h-20 rounded-2xl bg-[var(--color-bg-tertiary)] flex items-center justify-center mb-4">
                            <Image className="w-10 h-10 text-[var(--color-text-muted)]" />
                        </div>
                        <h3 className="text-lg font-medium text-[var(--color-text-primary)] mb-2">
                            No Character Selected
                        </h3>
                        <p className="text-sm text-[var(--color-text-secondary)] max-w-xs">
                            Select a character from the sidebar to view and manage their dataset images.
                        </p>
                    </div>
                )}
            </div>
        </main>
    );
}
