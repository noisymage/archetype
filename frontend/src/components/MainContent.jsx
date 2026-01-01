import { Filter, Grid3X3, LayoutList, Image as ImageIcon } from 'lucide-react';
import { cn } from '../lib/utils';
import { ImageCard } from './ImageCard';
import { Skeleton } from './ui/Skeleton';
import { Button } from './ui/Button';

// Mock image data
const mockImages = Array.from({ length: 12 }, (_, i) => ({
    id: i + 1,
    status: ['pending', 'analyzed', 'approved', 'rejected'][Math.floor(Math.random() * 4)],
    score: Math.random() * 0.4 + 0.6, // 0.6 - 1.0 range
}));

/**
 * Main content area with responsive image grid
 */
export function MainContent({ selectedCharacter, onSelectImage, selectedImage, isLoading }) {
    return (
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
                            ID: <span className="text-zinc-400">{selectedCharacter.id.toString().padStart(4, '0')}</span> • {selectedCharacter.imageCount} IMAGES
                        </p>
                    ) : (
                        <Skeleton className="h-4 w-32 mt-1 bg-zinc-800/50" />
                    )}
                </div>

                {selectedCharacter && (
                    <div className="flex items-center gap-3">
                        <Button variant="secondary" size="sm" className="gap-2 text-zinc-400 hover:text-white border-white/5 bg-white/5">
                            <Filter className="w-3.5 h-3.5" />
                            Filter
                        </Button>
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

            {/* Image Grid */}
            <div className="flex-1 overflow-y-auto p-8 z-10">
                {selectedCharacter ? (
                    isLoading ? (
                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
                            {Array.from({ length: 12 }).map((_, i) => (
                                <Skeleton key={i} className="aspect-square rounded-xl bg-zinc-800/40" />
                            ))}
                        </div>
                    ) : (
                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6 p-2">
                            {mockImages.map((image) => (
                                <ImageCard
                                    key={image.id}
                                    image={image}
                                    isSelected={selectedImage?.id === image.id}
                                    onClick={() => onSelectImage(image)}
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
    );
}
