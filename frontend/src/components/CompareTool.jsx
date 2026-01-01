import { useState, useRef } from 'react';
import { Upload, X, RefreshCw, AlertTriangle, CheckCircle, ChevronDown, ChevronRight } from 'lucide-react';
import { cn } from '../lib/utils';
import { Button } from './ui/Button';

/**
 * Image Analysis Result Viewer
 */
function AnalysisResult({ data, loading, error }) {
    const [expanded, setExpanded] = useState({ face: true, pose: true, body: true });

    const toggle = (section) => {
        setExpanded(prev => ({ ...prev, [section]: !prev[section] }));
    };

    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center h-64 text-zinc-500">
                <RefreshCw className="w-8 h-8 animate-spin mb-2" />
                <span className="text-sm">Analyzing...</span>
            </div>
        );
    }

    if (error) {
        return (
            <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-4 flex items-start gap-3 mt-4">
                <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <p className="text-sm text-red-200">{error}</p>
            </div>
        );
    }

    if (!data) return null;

    return (
        <div className="space-y-4 mt-4 text-sm font-mono overflow-y-auto max-h-[500px] bg-black/20 p-2 rounded-lg border border-white/5">
            {/* Face Section */}
            <div className="border border-white/10 rounded-md overflow-hidden bg-zinc-900/50">
                <button
                    onClick={() => toggle('face')}
                    className="w-full px-3 py-2 flex items-center justify-between hover:bg-white/5 transition-colors"
                >
                    <span className="font-semibold text-zinc-300">Face Analysis</span>
                    {expanded.face ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                </button>
                {expanded.face && (
                    <div className="p-3 border-t border-white/10 space-y-2">
                        {data.face ? (
                            <>
                                {data.face.error ? (
                                    <div className="text-red-400">Error: {data.face.error}</div>
                                ) : (
                                    <>
                                        <div className="flex justify-between">
                                            <span className="text-zinc-500">Confidence:</span>
                                            <span className="text-cyan-400">{data.face.confidence?.toFixed(4)}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-zinc-500">BBox:</span>
                                            <span className="text-zinc-300">[{data.face.bbox?.map(n => Math.round(n)).join(', ')}]</span>
                                        </div>
                                        {data.face.pose && (
                                            <div className="mt-2">
                                                <div className="text-zinc-500 mb-1">Head Pose:</div>
                                                <div className="grid grid-cols-3 gap-2 pl-2">
                                                    <div>
                                                        <span className="text-xs text-zinc-600 block">YAW</span>
                                                        <span className="text-zinc-300">{data.face.pose.yaw?.toFixed(1)}°</span>
                                                    </div>
                                                    <div>
                                                        <span className="text-xs text-zinc-600 block">PITCH</span>
                                                        <span className="text-zinc-300">{data.face.pose.pitch?.toFixed(1)}°</span>
                                                    </div>
                                                    <div>
                                                        <span className="text-xs text-zinc-600 block">ROLL</span>
                                                        <span className="text-zinc-300">{data.face.pose.roll?.toFixed(1)}°</span>
                                                    </div>
                                                </div>
                                            </div>
                                        )}
                                    </>
                                )}
                            </>
                        ) : (
                            <span className="text-zinc-500 italic">No face detected</span>
                        )}
                    </div>
                )}
            </div>

            {/* Pose Section */}
            <div className="border border-white/10 rounded-md overflow-hidden bg-zinc-900/50">
                <button
                    onClick={() => toggle('pose')}
                    className="w-full px-3 py-2 flex items-center justify-between hover:bg-white/5 transition-colors"
                >
                    <span className="font-semibold text-zinc-300">2D Pose (YOLO)</span>
                    {expanded.pose ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                </button>
                {expanded.pose && (
                    <div className="p-3 border-t border-white/10">
                        {data.pose ? (
                            <>
                                {data.pose.error ? (
                                    <div className="text-red-400">Error: {data.pose.error}</div>
                                ) : (
                                    <div className="space-y-2">
                                        <div className="flex justify-between">
                                            <span className="text-zinc-500">Confidence:</span>
                                            <span className="text-cyan-400">{data.pose.confidence?.toFixed(4)}</span>
                                        </div>
                                        <div className="text-zinc-500 mt-2">Keypoints ({Object.keys(data.pose.keypoints || {}).length}):</div>
                                        <div className="grid grid-cols-2 gap-x-4 gap-y-1 pl-2 text-xs">
                                            {Object.entries(data.pose.keypoints || {}).map(([key, val]) => (
                                                <div key={key} className="flex justify-between">
                                                    <span className="text-zinc-600">{key}:</span>
                                                    <span className={val.confidence > 0.5 ? "text-zinc-300" : "text-zinc-600"}>
                                                        {val.confidence?.toFixed(2)}
                                                    </span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </>
                        ) : (
                            <span className="text-zinc-500 italic">No pose detected</span>
                        )}
                    </div>
                )}
            </div>

            {/* Body/3D Section */}
            <div className="border border-white/10 rounded-md overflow-hidden bg-zinc-900/50">
                <button
                    onClick={() => toggle('body')}
                    className="w-full px-3 py-2 flex items-center justify-between hover:bg-white/5 transition-colors"
                >
                    <span className="font-semibold text-zinc-300">3D Body Analysis</span>
                    {expanded.body ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                </button>
                {expanded.body && (
                    <div className="p-3 border-t border-white/10">
                        {data.body ? (
                            <>
                                {data.body.error ? (
                                    <div className="text-red-400">Error: {data.body.error}</div>
                                ) : (
                                    <div className="space-y-2">
                                        <div className="flex justify-between">
                                            <span className="text-zinc-500">Mode:</span>
                                            <span className={data.body.degraded_mode ? "text-yellow-400" : "text-green-400"}>
                                                {data.body.degraded_mode ? "2D Degraded" : "3D Mesh"}
                                            </span>
                                        </div>
                                        {data.body.volume_estimate && (
                                            <div className="flex justify-between">
                                                <span className="text-zinc-500">Volume Est:</span>
                                                <span className="text-zinc-300">{data.body.volume_estimate.toFixed(2)} L</span>
                                            </div>
                                        )}
                                        {data.body.ratios && (
                                            <>
                                                <div className="text-zinc-500 mt-2">Ratios:</div>
                                                <div className="pl-2 space-y-1">
                                                    {Object.entries(data.body.ratios).map(([k, v]) => (
                                                        <div key={k} className="flex justify-between text-xs">
                                                            <span className="text-zinc-600">{k}:</span>
                                                            <span className="text-zinc-300">
                                                                {typeof v === 'number' ? v.toFixed(3) : String(v)}
                                                            </span>
                                                        </div>
                                                    ))}
                                                </div>
                                            </>
                                        )}
                                        {data.body.betas && (
                                            <div className="mt-2">
                                                <span className="text-zinc-500 text-xs">Betas (Shape Params):</span>
                                                <div className="text-[10px] text-zinc-600 break-all bg-black/30 p-1 rounded">
                                                    {JSON.stringify(data.body.betas.slice(0, 5))}...
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </>
                        ) : (
                            <span className="text-zinc-500 italic">No body analysis available</span>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

/**
 * Single Image Upload and Display Card
 */
function ImageSlot({ title, imageState, onFileSelect, onRemove }) {
    const inputRef = useRef(null);

    const handleDragOver = (e) => {
        e.preventDefault();
        e.stopPropagation();
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            onFileSelect(e.dataTransfer.files[0]);
        }
    };

    return (
        <div className="flex-1 flex flex-col h-full overflow-hidden">
            <div className="px-4 py-3 bg-zinc-900 border-b border-white/10 flex justify-between items-center">
                <h3 className="font-semibold text-zinc-200">{title}</h3>
                {imageState.file && (
                    <Button
                        variant="ghost"
                        size="icon"
                        onClick={onRemove}
                        className="h-6 w-6 text-zinc-500 hover:text-red-400"
                    >
                        <X className="w-4 h-4" />
                    </Button>
                )}
            </div>

            <div className="flex-1 overflow-y-auto p-4">
                {!imageState.file ? (
                    <div
                        className="h-64 border-2 border-dashed border-white/10 rounded-xl flex flex-col items-center justify-center cursor-pointer hover:border-cyan-500/50 hover:bg-white/5 transition-all text-zinc-500 hover:text-cyan-400"
                        onClick={() => inputRef.current?.click()}
                        onDragOver={handleDragOver}
                        onDrop={handleDrop}
                    >
                        <Upload className="w-10 h-10 mb-4 opacity-50" />
                        <p className="font-medium">Click or Drag Image Here</p>
                        <span className="text-xs mt-2 opacity-50">Supports PNG, JPG</span>
                        <input
                            ref={inputRef}
                            type="file"
                            accept="image/*"
                            onChange={(e) => {
                                if (e.target.files?.[0]) onFileSelect(e.target.files[0]);
                            }}
                            className="hidden"
                        />
                    </div>
                ) : (
                    <div className="space-y-4">
                        <div className="relative aspect-video bg-black rounded-lg overflow-hidden border border-white/10 flex items-center justify-center">
                            <img
                                src={imageState.preview}
                                alt={title}
                                className="max-w-full max-h-full object-contain"
                            />
                        </div>
                        <AnalysisResult
                            data={imageState.data}
                            loading={imageState.loading}
                            error={imageState.error}
                        />
                    </div>
                )}
            </div>
        </div>
    );
}

/**
 * Compare Tool Component
 */
export function CompareTool() {
    const [images, setImages] = useState({
        reference: { file: null, preview: null, data: null, loading: false },
        candidate: { file: null, preview: null, data: null, loading: false }
    });

    const handleFileSelect = async (type, file) => {
        // Create preview URL
        const previewUrl = URL.createObjectURL(file);

        // Update state
        setImages(prev => ({
            ...prev,
            [type]: { file, preview: previewUrl, data: null, loading: true, error: null }
        }));

        // Upload and analyze
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/tools/analyze_upload', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Analysis failed');
            }

            const data = await response.json();

            setImages(prev => ({
                ...prev,
                [type]: { ...prev[type], data, loading: false }
            }));

        } catch (error) {
            console.error("Upload error:", error);
            setImages(prev => ({
                ...prev,
                [type]: { ...prev[type], loading: false, error: error.message }
            }));
        }
    };

    const handleRemove = (type) => {
        setImages(prev => ({
            ...prev,
            [type]: { file: null, preview: null, data: null, loading: false }
        }));
    };

    return (
        <div className="flex-1 flex flex-col h-full bg-zinc-950 text-white overflow-hidden">
            {/* Header */}
            <div className="px-6 py-4 border-b border-white/5 bg-zinc-950/50 backdrop-blur-sm">
                <div>
                    <h2 className="text-xl font-semibold text-zinc-100 tracking-tight">
                        Image Comparison Tool
                    </h2>
                    <p className="text-xs text-zinc-500 mt-1 font-mono">
                        Upload two images to compare vision model outputs.
                    </p>
                </div>
            </div>

            <div className="flex-1 flex overflow-hidden">
                <div className="flex-1 flex border-r border-white/5">
                    <ImageSlot
                        title="Reference Image"
                        imageState={images.reference}
                        onFileSelect={(f) => handleFileSelect('reference', f)}
                        onRemove={() => handleRemove('reference')}
                    />
                </div>
                <div className="flex-1 flex">
                    <ImageSlot
                        title="Candidate Image"
                        imageState={images.candidate}
                        onFileSelect={(f) => handleFileSelect('candidate', f)}
                        onRemove={() => handleRemove('candidate')}
                    />
                </div>
            </div>
        </div>
    );
}
