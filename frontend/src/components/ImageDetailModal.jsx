import { useState, useRef, useEffect } from 'react';
import { X, ZoomIn, ZoomOut, Eye, EyeOff, Bone, User, Square, Maximize2 } from 'lucide-react';
import { cn } from '../lib/utils';
import { Button } from './ui/Button';
import { getFullImageUrl } from '../lib/api';

/**
 * Image Detail Modal - Full-screen view with overlay toggles
 */
export function ImageDetailModal({ image, metrics, onClose }) {
    const [showSkeleton, setShowSkeleton] = useState(false);
    const [showFaceBbox, setShowFaceBbox] = useState(false);
    const [ratiosView, setRatiosView] = useState('preferred'); // 'preferred', '3d', or '2d'
    const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
    const canvasRef = useRef(null);
    const imgRef = useRef(null);

    if (!image) return null;

    const imageUrl = getFullImageUrl(image.original_path || image.path);

    // Redraw overlays when toggles change or image size changes
    useEffect(() => {
        const canvas = canvasRef.current;
        const img = imgRef.current;
        if (!canvas || !img || !imageSize.width) return;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Calculate scale from original image to displayed size
        const scaleX = img.clientWidth / imageSize.width;
        const scaleY = img.clientHeight / imageSize.height;

        // Draw face bounding box
        if (showFaceBbox && metrics?.face_bbox) {
            const [x1, y1, x2, y2] = metrics.face_bbox;
            ctx.strokeStyle = '#22d3ee'; // cyan-400
            ctx.lineWidth = 2;
            ctx.strokeRect(
                x1 * scaleX,
                y1 * scaleY,
                (x2 - x1) * scaleX,
                (y2 - y1) * scaleY
            );
        }

        // Draw skeleton keypoints
        if (showSkeleton && metrics?.keypoints) {
            const kp = metrics.keypoints;

            // Define connections for skeleton
            const connections = [
                ['left_shoulder', 'right_shoulder'],
                ['left_shoulder', 'left_elbow'],
                ['left_elbow', 'left_wrist'],
                ['right_shoulder', 'right_elbow'],
                ['right_elbow', 'right_wrist'],
                ['left_shoulder', 'left_hip'],
                ['right_shoulder', 'right_hip'],
                ['left_hip', 'right_hip'],
                ['left_hip', 'left_knee'],
                ['left_knee', 'left_ankle'],
                ['right_hip', 'right_knee'],
                ['right_knee', 'right_ankle'],
                ['nose', 'left_eye'],
                ['nose', 'right_eye'],
                ['left_eye', 'left_ear'],
                ['right_eye', 'right_ear']
            ];

            // Draw connections
            ctx.strokeStyle = '#a855f7'; // purple-500
            ctx.lineWidth = 2;
            for (const [from, to] of connections) {
                if (kp[from] && kp[to] && kp[from].confidence > 0.3 && kp[to].confidence > 0.3) {
                    ctx.beginPath();
                    ctx.moveTo(kp[from].x * scaleX, kp[from].y * scaleY);
                    ctx.lineTo(kp[to].x * scaleX, kp[to].y * scaleY);
                    ctx.stroke();
                }
            }

            // Draw keypoints
            ctx.fillStyle = '#f472b6'; // pink-400
            for (const [name, point] of Object.entries(kp)) {
                if (point.confidence > 0.3) {
                    ctx.beginPath();
                    ctx.arc(point.x * scaleX, point.y * scaleY, 4, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
        }
    }, [showSkeleton, showFaceBbox, metrics, imageSize]);

    const handleImageLoad = (e) => {
        setImageSize({
            width: e.target.naturalWidth,
            height: e.target.naturalHeight
        });
    };

    const getStatusColor = (status) => {
        switch (status) {
            case 'approved': return 'text-green-400';
            case 'rejected': return 'text-red-400';
            case 'analyzed': return 'text-yellow-400';
            default: return 'text-zinc-400';
        }
    };

    const getSimilarityColor = (score) => {
        if (score === null || score === undefined) return 'text-zinc-400';
        if (score >= 0.85) return 'text-green-400';
        if (score >= 0.7) return 'text-yellow-400';
        return 'text-red-400';
    };

    // Parse dual metrics structure
    const limbRatiosData = image.limb_ratios || {};
    const metrics3d = limbRatiosData.metrics_3d || null;
    const metrics2d = limbRatiosData.metrics_2d || null;
    const preferred = limbRatiosData.preferred || 'none';

    // Backward compatibility: old single-metric format
    let displayedRatios = {};
    let analysisMethod = 'Unknown';

    if (metrics3d || metrics2d) {
        // New dual-metrics structure
        if (ratiosView === '3d' && metrics3d) {
            displayedRatios = metrics3d.ratios || {};
            analysisMethod = '3D Mesh (SMPLer-X)';
        } else if (ratiosView === '2d' && metrics2d) {
            displayedRatios = metrics2d.ratios || {};
            analysisMethod = '2D Keypoints (YOLO)';
        } else if (ratiosView === 'preferred') {
            // Use preferred method
            if (preferred === '3d' && metrics3d) {
                displayedRatios = metrics3d.ratios || {};
                analysisMethod = '3D Mesh (SMPLer-X)';
            } else if (preferred === '2d' && metrics2d) {
                displayedRatios = metrics2d.ratios || {};
                analysisMethod = '2D Keypoints (YOLO)';
            }
        }
    } else if (limbRatiosData.ratios) {
        // Old structure: { ratios: {...}, degraded_mode: bool }
        analysisMethod = limbRatiosData.degraded_mode ? '2D Keypoints (YOLO)' : '3D Mesh (SMPLer-X)';
        displayedRatios = limbRatiosData.ratios;
    }

    return (
        <div className="fixed inset-0 z-50 flex">
            {/* Backdrop */}
            <div className="absolute inset-0 bg-black/90" onClick={onClose} />

            {/* Modal Container */}
            <div className="relative flex w-full h-full">
                {/* Main Image Area */}
                <div className="flex-1 flex items-center justify-center p-8 relative">
                    {/* Close button */}
                    <button
                        onClick={onClose}
                        className="absolute top-4 right-4 p-2 rounded-lg bg-black/50 hover:bg-black/70 text-white transition-colors z-10"
                    >
                        <X className="w-6 h-6" />
                    </button>

                    {/* Overlay Toggle Buttons */}
                    <div className="absolute top-4 left-4 flex gap-2 z-10">
                        <Button
                            variant={showFaceBbox ? "primary" : "secondary"}
                            size="sm"
                            onClick={() => setShowFaceBbox(!showFaceBbox)}
                            className="gap-2"
                            disabled={!metrics?.face_bbox}
                        >
                            <Square className="w-4 h-4" />
                            Face
                        </Button>
                        <Button
                            variant={showSkeleton ? "primary" : "secondary"}
                            size="sm"
                            onClick={() => setShowSkeleton(!showSkeleton)}
                            className="gap-2"
                            disabled={!metrics?.keypoints}
                        >
                            <Bone className="w-4 h-4" />
                            Skeleton
                        </Button>
                    </div>

                    {/* Image with Canvas Overlay */}
                    <div className="relative max-w-full max-h-full">
                        <img
                            ref={imgRef}
                            src={imageUrl}
                            alt={image.filename || 'Dataset image'}
                            onLoad={handleImageLoad}
                            className="max-w-full max-h-[80vh] object-contain rounded-lg"
                        />
                        <canvas
                            ref={canvasRef}
                            width={imgRef.current?.clientWidth || 800}
                            height={imgRef.current?.clientHeight || 600}
                            className="absolute inset-0 pointer-events-none"
                        />
                    </div>
                </div>

                {/* Metrics Panel */}
                <div className="w-80 bg-zinc-950 border-l border-white/10 p-6 overflow-y-auto">
                    <h3 className="text-lg font-semibold text-white mb-6">Image Analysis</h3>

                    {/* Filename */}
                    <div className="mb-6">
                        <span className="text-xs text-zinc-500 uppercase tracking-wider">Filename</span>
                        <p className="text-sm text-white font-mono mt-1 break-all">
                            {image.filename || image.original_path?.split('/').pop()}
                        </p>
                    </div>

                    {/* Status */}
                    <div className="mb-6">
                        <span className="text-xs text-zinc-500 uppercase tracking-wider">Status</span>
                        <p className={cn("text-sm font-medium mt-1 capitalize", getStatusColor(image.status))}>
                            {image.status || 'Pending'}
                        </p>
                    </div>

                    {/* Shot Type */}
                    {metrics?.shot_type && (
                        <div className="mb-6">
                            <span className="text-xs text-zinc-500 uppercase tracking-wider">Shot Type</span>
                            <p className="text-sm text-white font-medium mt-1 capitalize">
                                {metrics.shot_type}
                            </p>
                        </div>
                    )}

                    {/* Analysis Method */}
                    {analysisMethod && (
                        <div className="mb-6">
                            <span className="text-xs text-zinc-500 uppercase tracking-wider">Method</span>
                            <p className={cn("text-sm font-medium mt-1",
                                analysisMethod.includes('SMPL') ? "text-fuchsia-400" : "text-yellow-400"
                            )}>
                                {analysisMethod}
                            </p>
                        </div>
                    )}

                    {/* Scores Section */}
                    <div className="border-t border-white/10 pt-6 mt-6">
                        <h4 className="text-sm font-medium text-zinc-300 mb-4">Scores</h4>

                        {/* Face Similarity */}
                        <div className="mb-4">
                            <div className="flex items-center justify-between mb-1">
                                <div className="flex items-center gap-2">
                                    <span className="text-xs text-zinc-400">Face Similarity</span>
                                    {image.closest_face_ref && (
                                        <div className="relative group z-10">
                                            <div className="w-16 h-16 rounded-full overflow-hidden border border-white/10 ring-1 ring-black/50 cursor-help transition-transform hover:scale-[2.5] hover:z-50 origin-left shadow-lg">
                                                <img
                                                    src={getFullImageUrl(image.closest_face_ref)}
                                                    alt="Closest reference"
                                                    className="w-full h-full object-cover"
                                                />
                                            </div>
                                            {/* Tooltip */}
                                            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-3 w-max max-w-[150px] px-2 py-1 bg-black/90 border border-white/10 text-[10px] text-zinc-300 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                                                <p className="truncate">Match: {image.closest_face_ref.split(/[/\\]/).pop()}</p>
                                            </div>
                                        </div>
                                    )}
                                </div>
                                <span className={cn("text-sm font-mono", getSimilarityColor(image.face_similarity))}>
                                    {image.face_similarity !== null && image.face_similarity !== undefined
                                        ? `${(image.face_similarity * 100).toFixed(1)}%`
                                        : '—'
                                    }
                                </span>
                            </div>
                            <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                                <div
                                    className={cn(
                                        "h-full rounded-full transition-all",
                                        image.face_similarity >= 0.85 ? "bg-green-500" :
                                            image.face_similarity >= 0.7 ? "bg-yellow-500" : "bg-red-500"
                                    )}
                                    style={{ width: `${(image.face_similarity || 0) * 100}%` }}
                                />
                            </div>
                        </div>


                        {/* Body Consistency (3D) */}
                        {metrics3d && (
                            <div className="mb-4">
                                <div className="flex items-center justify-between mb-1">
                                    <span className="text-xs text-zinc-400">Body Consistency (3D)</span>
                                    <span className="text-sm font-mono text-cyan-400">
                                        {(metrics3d.consistency_score * 100).toFixed(1)}%
                                    </span>
                                </div>
                                <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-cyan-500 rounded-full transition-all"
                                        style={{ width: `${metrics3d.consistency_score * 100}%` }}
                                    />
                                </div>
                            </div>
                        )}

                        {/* Body Consistency (2D) */}
                        {metrics2d && (
                            <div className="mb-4">
                                <div className="flex items-center justify-between mb-1">
                                    <span className="text-xs text-zinc-400">Body Consistency (2D)</span>
                                    <span className="text-sm font-mono text-fuchsia-400">
                                        {(metrics2d.consistency_score * 100).toFixed(1)}%
                                    </span>
                                </div>
                                <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-fuchsia-500 rounded-full transition-all"
                                        style={{ width: `${metrics2d.consistency_score * 100}%` }}
                                    />
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Limb Ratios (if available) */}
                    {displayedRatios && Object.keys(displayedRatios).length > 0 && (
                        <div className="border-t border-white/10 pt-6 mt-6">
                            <div className="flex items-center justify-between mb-4">
                                <h4 className="text-sm font-medium text-zinc-300">Limb Ratios</h4>

                                {/* View Toggle (only show if both metrics exist) */}
                                {metrics3d && metrics2d && (
                                    <div className="flex gap-1 bg-zinc-900 rounded-md p-1">
                                        <button
                                            onClick={() => setRatiosView('3d')}
                                            className={cn(
                                                "px-2 py-1 text-xs rounded transition-colors",
                                                ratiosView === '3d' || (ratiosView === 'preferred' && preferred === '3d')
                                                    ? "bg-cyan-500/20 text-cyan-400"
                                                    : "text-zinc-500 hover:text-zinc-300"
                                            )}
                                        >
                                            3D
                                        </button>
                                        <button
                                            onClick={() => setRatiosView('2d')}
                                            className={cn(
                                                "px-2 py-1 text-xs rounded transition-colors",
                                                ratiosView === '2d' || (ratiosView === 'preferred' && preferred === '2d')
                                                    ? "bg-fuchsia-500/20 text-fuchsia-400"
                                                    : "text-zinc-500 hover:text-zinc-300"
                                            )}
                                        >
                                            2D
                                        </button>
                                    </div>
                                )}
                            </div>
                            <div className="space-y-2">
                                {Object.entries(displayedRatios).map(([key, value]) => (
                                    <div key={key} className="flex items-center justify-between">
                                        <span className="text-xs text-zinc-400 capitalize">
                                            {key.replace(/_/g, ' ')}
                                        </span>
                                        <span className="text-sm font-mono text-zinc-300">
                                            {typeof value === 'number' ? value.toFixed(3) : value}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Actions */}
                    <div className="border-t border-white/10 pt-6 mt-6 space-y-2">
                        <Button variant="secondary" className="w-full gap-2">
                            <Eye className="w-4 h-4" />
                            Compare with Reference
                        </Button>
                    </div>
                </div>
            </div>
        </div>
    );
}
