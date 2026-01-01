import { Image as ImageIcon, CheckCircle, XCircle, Clock, AlertCircle } from 'lucide-react';
import { cn } from '../lib/utils';
import { getThumbnailUrl } from '../lib/api';

const statusConfig = {
    pending: { color: 'bg-yellow-500', glow: 'shadow-yellow-500/50', border: 'border-yellow-500/50' },
    analyzed: { color: 'bg-blue-500', glow: 'shadow-blue-500/50', border: 'border-blue-500/50' },
    approved: { color: 'bg-emerald-500', glow: 'shadow-emerald-500/50', border: 'border-emerald-500/50' },
    rejected: { color: 'bg-red-500', glow: 'shadow-red-500/50', border: 'border-red-500/50' },
    error: { color: 'bg-orange-500', glow: 'shadow-orange-500/50', border: 'border-orange-500/50' },
};

export function ImageCard({ image, onClick, isSelected }) {
    const status = statusConfig[image.status] || statusConfig.pending;
    const score = image.face_similarity || image.score || 0;
    const thumbnailUrl = image.original_path ? getThumbnailUrl(image.original_path) : null;

    return (
        <button
            onClick={onClick}
            className={cn(
                'group relative w-full aspect-square rounded-xl overflow-hidden transition-all duration-300 ease-out',
                'bg-zinc-900',
                isSelected
                    ? 'ring-2 ring-cyan-500 ring-offset-2 ring-offset-zinc-950 shadow-[0_0_20px_rgba(6,182,212,0.3)] z-10 scale-[1.02]'
                    : 'hover:scale-[1.05] hover:z-10',
                // Color-coded border based on status
                !isSelected && `border-2 ${status.border}`
            )}
        >
            {/* Thumbnail or Placeholder */}
            <div className="absolute inset-0 bg-gradient-to-br from-zinc-800 to-zinc-900 flex items-center justify-center">
                {thumbnailUrl ? (
                    <img
                        src={thumbnailUrl}
                        alt={image.filename || 'Image'}
                        className="w-full h-full object-cover"
                        loading="lazy"
                    />
                ) : (
                    <ImageIcon className="w-8 h-8 text-zinc-700 group-hover:text-zinc-600 transition-colors" />
                )}
            </div>

            {/* Status Indicator (Glowing Dot) */}
            <div className="absolute top-3 right-3">
                <div className={cn("w-2.5 h-2.5 rounded-full", status.color, status.glow, "shadow-lg")} />
            </div>

            {/* Shot Type Badge */}
            {image.shot_type && (
                <div className="absolute top-3 left-3">
                    <span className="px-2 py-0.5 text-[10px] font-medium bg-black/60 text-zinc-300 rounded-full capitalize">
                        {image.shot_type}
                    </span>
                </div>
            )}

            {/* Metadata Overlay (Slide Up) */}
            <div className="absolute inset-x-0 bottom-0 p-3 bg-gradient-to-t from-black/90 via-black/60 to-transparent translate-y-full group-hover:translate-y-0 transition-transform duration-300">
                <div className="flex flex-col gap-1 items-start">
                    {/* Filename */}
                    <span className="text-xs text-zinc-300 font-medium truncate w-full">
                        {image.filename || 'Unknown'}
                    </span>

                    {/* Score */}
                    {score > 0 && (
                        <>
                            <div className="flex items-center justify-between w-full text-[10px] text-zinc-400 font-mono uppercase tracking-wider">
                                <span>Similarity</span>
                                <span className={cn(
                                    "text-zinc-200",
                                    score >= 0.85 ? "text-green-400" :
                                        score >= 0.7 ? "text-yellow-400" : "text-red-400"
                                )}>
                                    {(score * 100).toFixed(0)}%
                                </span>
                            </div>
                            <div className="w-full h-1 bg-white/10 rounded-full overflow-hidden">
                                <div
                                    className={cn(
                                        "h-full rounded-full",
                                        score >= 0.85 ? "bg-green-500" :
                                            score >= 0.7 ? "bg-yellow-500" : "bg-red-500"
                                    )}
                                    style={{ width: `${score * 100}%` }}
                                />
                            </div>
                        </>
                    )}

                    <div className="flex items-center gap-1.5 mt-1 text-xs text-zinc-300 font-medium capitalize">
                        <span className={cn("w-1.5 h-1.5 rounded-full inline-block", status.color)} />
                        {image.status}
                    </div>
                </div>
            </div>

            {/* Border */}
            {!isSelected && (
                <div className="absolute inset-0 rounded-xl border border-white/5 group-hover:border-white/20 transition-colors pointer-events-none" />
            )}
        </button>
    );
}
