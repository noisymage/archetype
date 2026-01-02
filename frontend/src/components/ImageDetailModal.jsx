import { useState, useRef, useEffect } from 'react';
import { X, ZoomIn, ZoomOut, Eye, EyeOff, Bone, User, Square, Maximize2, RefreshCw, Split, ChevronDown } from 'lucide-react';
import { cn } from '../lib/utils';
import { Button } from './ui/Button';
import { getFullImageUrl, reprocessImage, getThumbnailUrl } from '../lib/api';
import { HEAD_SLOTS, BODY_SLOTS } from '../lib/constants';

/**
 * Image Detail Modal - Full-screen view with overlay toggles and comparison mode
 */
export function ImageDetailModal({ image, metrics, onClose, onUpdate, references }) {
    const [showSkeleton, setShowSkeleton] = useState(false);
    const [showFaceBbox, setShowFaceBbox] = useState(false);
    const [isReprocessing, setIsReprocessing] = useState(false);
    const [ratiosView, setRatiosView] = useState('preferred'); // 'preferred', '3d', or '2d'
    const [imageSize, setImageSize] = useState({ width: 0, height: 0 });

    // Comparison Mode State
    const [compareMode, setCompareMode] = useState(false);
    const [selectedRefKey, setSelectedRefKey] = useState(null);

    const canvasRef = useRef(null);
    const imgRef = useRef(null);
    const refImgRef = useRef(null);

    // Auto-select reference based on closest match or shot_type
    useEffect(() => {
        if (!references || selectedRefKey) return;

        // 1. First priority: Use the actual closest reference calculated by backend
        if (image.closest_face_ref) {
            // Reverse lookup: find key where path matches closest_face_ref
            const matchedEntry = Object.entries(references).find(([_, path]) => {
                // simple normalization to handle potential path differences
                return path.endsWith(image.closest_face_ref.split('/').pop());
            });

            if (matchedEntry) {
                console.log('Using closest face ref:', matchedEntry[0]);
                setSelectedRefKey(matchedEntry[0]);
                return;
            }
        }

        // 2. Second priority: Smart mapping based on shot_type
        if (metrics?.shot_type) {
            const shotType = metrics.shot_type.toLowerCase();
            let targetRef = null;

            // Precise mapping
            if (shotType === 'close-up') targetRef = 'head_front';
            else if (shotType === 'medium') targetRef = 'body_front';
            else if (shotType === 'full-body') targetRef = 'body_front';

            // Fallback: Check for exact match (e.g. if shot_type matches slot key)
            if (!targetRef && references[shotType]) targetRef = shotType;

            if (targetRef && references[targetRef]) {
                setSelectedRefKey(targetRef);
                return;
            }
        }

        // 3. Last resort: default to body_front or head_front if available
        if (references['body_front']) {
            setSelectedRefKey('body_front');
        } else if (references['head_front']) {
            setSelectedRefKey('head_front');
        }
    }, [metrics, references, image]);

    if (!image) return null;

    const imageUrl = getFullImageUrl(image.original_path || image.path);
    const refUrl = selectedRefKey && references?.[selectedRefKey]
        ? getThumbnailUrl(references[selectedRefKey], 1024)
        : null;

    // Redraw overlays
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
            drawSkeleton(ctx, metrics.keypoints, scaleX, scaleY);
        }
    }, [showSkeleton, showFaceBbox, metrics, imageSize, compareMode]); // Re-draw on resize/compare mode toggle

    const drawSkeleton = (ctx, kp, scaleX, scaleY) => {
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
    };

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

    // Combine slots for selector
    const allSlots = [...HEAD_SLOTS, ...BODY_SLOTS];

    return (
        <div className="fixed inset-0 z-50 flex">
            {/* Backdrop */}
            <div className="absolute inset-0 bg-black/90" onClick={onClose} />

            {/* Modal Container */}
            <div className="relative flex w-full h-full pointer-events-none">
                {/* Main Image Area */}
                <div className="flex-1 flex flex-col relative overflow-hidden bg-black/50 pointer-events-auto">
                    {/* Close button - Moved to overlay to not conflict with split view */}
                    <button
                        onClick={onClose}
                        className="absolute top-4 right-4 p-2 rounded-lg bg-black/50 hover:bg-black/70 text-white transition-colors z-30"
                    >
                        <X className="w-6 h-6" />
                    </button>

                    {/* Toolbar */}
                    <div className="absolute top-4 left-4 flex gap-2 z-30">
                        <div className="bg-black/60 backdrop-blur-sm rounded-lg p-1 flex gap-1 border border-white/10">
                            <Button
                                variant={showFaceBbox ? "primary" : "ghost"}
                                size="sm"
                                onClick={() => setShowFaceBbox(!showFaceBbox)}
                                className="h-8 w-8 p-0"
                                title="Toggle Face Box"
                                disabled={!metrics?.face_bbox}
                            >
                                <Square className="w-4 h-4" />
                            </Button>
                            <Button
                                variant={showSkeleton ? "primary" : "ghost"}
                                size="sm"
                                onClick={() => setShowSkeleton(!showSkeleton)}
                                className="h-8 w-8 p-0"
                                title="Toggle Skeleton"
                                disabled={!metrics?.keypoints}
                            >
                                <Bone className="w-4 h-4" />
                            </Button>
                        </div>

                        {/* Comparison Toggle */}
                        {references && (
                            <div className="bg-black/60 backdrop-blur-sm rounded-lg p-1 border border-white/10">
                                <Button
                                    variant={compareMode ? "primary" : "ghost"}
                                    size="sm"
                                    onClick={() => setCompareMode(!compareMode)}
                                    className="gap-2 h-8 px-3"
                                >
                                    <Split className="w-4 h-4" />
                                    <span className="text-xs font-medium">Compare</span>
                                </Button>
                            </div>
                        )}
                    </div>

                    <div className="flex-1 flex items-stretch">
                        {/* Left Pane: Dataset Image */}
                        <div className={cn(
                            "flex-1 flex items-center justify-center p-8 relative transition-all duration-300",
                            compareMode ? "w-1/2 border-r border-white/10" : "w-full"
                        )}>
                            <div className="relative max-w-full max-h-full">
                                <img
                                    ref={imgRef}
                                    src={imageUrl}
                                    alt={image.filename || 'Dataset image'}
                                    onLoad={handleImageLoad}
                                    className="max-w-full max-h-[calc(100vh-100px)] object-contain rounded-lg shadow-2xl"
                                />
                                <canvas
                                    ref={canvasRef}
                                    width={imgRef.current?.clientWidth || 800}
                                    height={imgRef.current?.clientHeight || 600}
                                    className="absolute inset-0 pointer-events-none"
                                />
                                {/* Label for split view */}
                                {compareMode && (
                                    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 px-3 py-1 bg-black/60 backdrop-blur rounded-full border border-white/10 text-xs text-zinc-300 font-medium z-20">
                                        Dataset Image
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Right Pane: Reference Image */}
                        {compareMode && (
                            <div className="flex-1 bg-zinc-950/50 flex flex-col animate-[fadeIn_0.3s_ease-out]">
                                {/* Reference Selector */}
                                <div className="p-4 border-b border-white/10 flex justify-center sticky top-0 bg-transparent z-20">
                                    <div className="relative inline-block w-64">
                                        <select
                                            value={selectedRefKey || ''}
                                            onChange={(e) => setSelectedRefKey(e.target.value)}
                                            className="w-full appearance-none bg-zinc-900 border border-white/10 hover:border-white/20 text-white text-sm rounded-lg px-4 py-2 pr-10 focus:outline-none focus:border-cyan-500/50 cursor-pointer"
                                        >
                                            <option value="" disabled>Select Reference...</option>
                                            {allSlots.filter(s => references && references[s.key]).map(slot => (
                                                <option key={slot.key} value={slot.key}>
                                                    {slot.icon} {slot.label}
                                                </option>
                                            ))}
                                        </select>
                                        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500 pointer-events-none" />
                                    </div>
                                </div>

                                <div className="flex-1 flex items-center justify-center p-8 relative overflow-hidden">
                                    {refUrl ? (
                                        <div className="relative max-w-full max-h-full">
                                            <img
                                                src={refUrl}
                                                alt="Reference"
                                                className="max-w-full max-h-[calc(100vh-160px)] object-contain rounded-lg shadow-2xl opacity-90"
                                            />
                                            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 px-3 py-1 bg-cyan-950/60 backdrop-blur rounded-full border border-cyan-500/30 text-xs text-cyan-200 font-medium z-20">
                                                Reference: {HEAD_SLOTS.find(s => s.key === selectedRefKey)?.label || BODY_SLOTS.find(s => s.key === selectedRefKey)?.label || selectedRefKey}
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="text-center text-zinc-500">
                                            <User className="w-12 h-12 mx-auto mb-3 opacity-20" />
                                            <p>No reference image selected</p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Metrics Panel */}
                <div className="w-80 bg-zinc-950 border-l border-white/10 p-6 overflow-y-auto shrink-0 z-20 shadow-xl pointer-events-auto">
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
                            <div className="flex items-center justify-between mt-1">
                                <p className="text-sm text-white font-medium capitalize">
                                    {metrics.shot_type}
                                </p>
                                {/* Quick compare button if not in compare mode */}
                                {!compareMode && references && references[metrics.shot_type] && (
                                    <button
                                        onClick={() => {
                                            setCompareMode(true);
                                            setSelectedRefKey(metrics.shot_type);
                                        }}
                                        className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1"
                                    >
                                        <Eye className="w-3 h-3" /> Compare
                                    </button>
                                )}
                            </div>
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
                                        </div>
                                    )}
                                </div>
                                <span className={cn("text-sm font-mono", getSimilarityColor(image.face_similarity))}>
                                    {image.face_similarity !== null && image.face_similarity !== undefined
                                        ? `${(image.face_similarity * 100).toFixed(1)}%`
                                        : '—'
                                    }
                                    {image.face_model_used && (
                                        <span className="ml-1 text-xs text-zinc-500 font-normal">
                                            via {image.face_model_used === 'adaface' ? 'AdaFace' : 'InsightFace'}
                                        </span>
                                    )}
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

                                {/* View Toggle */}
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
                        <Button
                            variant="secondary"
                            className="w-full gap-2"
                            onClick={() => {
                                setCompareMode(!compareMode);
                                if (!compareMode && metrics?.shot_type && references && references[metrics.shot_type] && !selectedRefKey) {
                                    setSelectedRefKey(metrics.shot_type);
                                }
                            }}
                        >
                            <Split className="w-4 h-4" />
                            {compareMode ? 'Exit Comparison' : 'Compare with Reference'}
                        </Button>
                        <Button
                            variant="secondary"
                            className="w-full gap-2"
                            onClick={async () => {
                                try {
                                    setIsReprocessing(true);
                                    const updatedImage = await reprocessImage(image.id);
                                    if (onUpdate) onUpdate(updatedImage);
                                } catch (e) {
                                    console.error(e);
                                    alert("Failed to reprocess image: " + e.message);
                                } finally {
                                    setIsReprocessing(false);
                                }
                            }}
                            disabled={isReprocessing}
                        >
                            <RefreshCw className={cn("w-4 h-4", isReprocessing && "animate-spin")} />
                            {isReprocessing ? "Reprocessing..." : "Reprocess Image"}
                        </Button>
                    </div>
                </div>
            </div>
        </div>
    );
}
