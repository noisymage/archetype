import { Image as ImageIcon, CheckCircle, XCircle, Clock, AlertCircle } from 'lucide-react';
import { cn } from '../lib/utils';

const statusConfig = {
    pending: { color: 'bg-yellow-500', glow: 'shadow-yellow-500/50' },
    analyzed: { color: 'bg-blue-500', glow: 'shadow-blue-500/50' },
    approved: { color: 'bg-emerald-500', glow: 'shadow-emerald-500/50' },
    rejected: { color: 'bg-red-500', glow: 'shadow-red-500/50' },
    error: { color: 'bg-orange-500', glow: 'shadow-orange-500/50' },
};

export function ImageCard({ image, onClick, isSelected }) {
    const status = statusConfig[image.status] || statusConfig.pending;

    return (
        <button
            onClick={onClick}
            className={cn(
                'group relative w-full aspect-square rounded-xl overflow-hidden transition-all duration-300 ease-out',
                'bg-zinc-900', // Base background
                isSelected
                    ? 'ring-2 ring-cyan-500 ring-offset-2 ring-offset-zinc-950 shadow-[0_0_20px_rgba(6,182,212,0.3)] z-10 scale-[1.02]'
                    : 'hover:scale-[1.05] hover:z-10'
            )}
        >
            {/* Placeholder / Image Logic */}
            <div className="absolute inset-0 bg-gradient-to-br from-zinc-800 to-zinc-900 flex items-center justify-center">
                {/* 
                  TODO: Replace this with actual <img> tag when data available.
                  <img src={image.url} className="w-full h-full object-cover" /> 
                */}
                <ImageIcon className="w-8 h-8 text-zinc-700 group-hover:text-zinc-600 transition-colors" />
            </div>

            {/* Status Indicator (Glowing Dot) */}
            <div className="absolute top-3 right-3">
                <div className={cn("w-2 h-2 rounded-full", status.color, status.glow, "shadow-lg")} />
            </div>

            {/* Metadata Overlay (Slide Up) */}
            <div className="absolute inset-x-0 bottom-0 p-3 bg-gradient-to-t from-black/90 via-black/60 to-transparent translate-y-full group-hover:translate-y-0 transition-transform duration-300">
                <div className="flex flex-col gap-1 items-start">
                    <div className="flex items-center justify-between w-full text-[10px] text-zinc-400 font-mono uppercase tracking-wider">
                        <span>Score</span>
                        <span className="text-zinc-200">{(image.score * 100).toFixed(0)}%</span>
                    </div>
                    {/* Progress Bar for Score */}
                    <div className="w-full h-1 bg-white/10 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-gradient-to-r from-cyan-500 to-fuchsia-500"
                            style={{ width: `${image.score * 100}%` }}
                        />
                    </div>

                    <div className="flex items-center gap-1.5 mt-1 text-xs text-zinc-300 font-medium capitalize">
                        <span className={cn("w-1.5 h-1.5 rounded-full inline-block", status.color)} />
                        {image.status}
                    </div>
                </div>
            </div>

            {/* Active Border Gradient (Optional subtle upgrade) */}
            {!isSelected && (
                <div className="absolute inset-0 rounded-xl border border-white/5 group-hover:border-white/20 transition-colors pointer-events-none" />
            )}
        </button>
    );
}
