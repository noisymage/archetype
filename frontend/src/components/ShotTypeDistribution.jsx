import { useMemo } from 'react';
import { cn } from '../lib/utils';

/**
 * Shot type distribution visualization component.
 * Shows a breakdown of image shot types (close-up, medium, full-body).
 */
export function ShotTypeDistribution({ images }) {
    const distribution = useMemo(() => {
        const counts = {
            'close-up': 0,
            'medium': 0,
            'full-body': 0,
            'unknown': 0
        };

        images.forEach(img => {
            const type = img.shot_type?.toLowerCase() || 'unknown';
            if (type in counts) {
                counts[type]++;
            } else {
                counts['unknown']++;
            }
        });

        const total = images.length || 1;

        return Object.entries(counts).map(([type, count]) => ({
            type,
            count,
            percentage: Math.round((count / total) * 100),
            color: getColorForType(type)
        })).filter(d => d.count > 0);
    }, [images]);

    if (images.length === 0) return null;

    return (
        <div className="flex items-center gap-4 px-4 py-2 bg-zinc-900/30 rounded-lg border border-white/5">
            <span className="text-xs text-zinc-500 uppercase tracking-wider font-medium">
                Shot Types
            </span>

            {/* Visual bar */}
            <div className="flex-1 flex h-2 rounded-full overflow-hidden bg-zinc-800 min-w-[120px] max-w-[200px]">
                {distribution.map((d, i) => (
                    <div
                        key={d.type}
                        className={cn("h-full transition-all", d.color)}
                        style={{ width: `${d.percentage}%` }}
                        title={`${d.type}: ${d.count} (${d.percentage}%)`}
                    />
                ))}
            </div>

            {/* Legend */}
            <div className="flex items-center gap-3">
                {distribution.map(d => (
                    <div key={d.type} className="flex items-center gap-1.5">
                        <div className={cn("w-2 h-2 rounded-full", d.color)} />
                        <span className="text-[10px] text-zinc-400 uppercase">
                            {d.type === 'close-up' ? 'CU' : d.type === 'full-body' ? 'FB' : d.type === 'medium' ? 'M' : '?'}
                        </span>
                        <span className="text-[10px] text-zinc-500">
                            {d.count}
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
}

function getColorForType(type) {
    switch (type) {
        case 'close-up':
            return 'bg-cyan-500';
        case 'medium':
            return 'bg-fuchsia-500';
        case 'full-body':
            return 'bg-amber-500';
        default:
            return 'bg-zinc-600';
    }
}
