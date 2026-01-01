import { useState } from 'react';
import { BarChart3, MessageSquare, Settings, X, Sparkles, Sliders, ChevronLeft } from 'lucide-react';
import { cn } from '../lib/utils';
import { Button } from './ui/Button';

const tabs = [
    { id: 'analysis', label: 'Analysis', icon: BarChart3 },
    { id: 'captions', label: 'Captions', icon: MessageSquare },
    { id: 'settings', label: 'Settings', icon: Settings },
];

/**
 * Right panel with tabbed interface
 */
export function DetailsPanel({ selectedImage, onClose }) {
    const [activeTab, setActiveTab] = useState('analysis');

    if (!selectedImage) {
        return (
            <aside className="w-80 h-full border-l border-white/5 bg-zinc-950/80 backdrop-blur-md flex flex-col items-center justify-center p-6 text-center z-20">
                <div className="w-16 h-16 rounded-2xl bg-zinc-900/50 border border-white/5 flex items-center justify-center mb-4">
                    <Sparkles className="w-8 h-8 text-cyan-500/50" />
                </div>
                <h3 className="text-base font-medium text-zinc-100 mb-2">
                    Select an Image
                </h3>
                <p className="text-sm text-zinc-500">
                    Click on an image to view its analysis, captions, and settings.
                </p>
            </aside>
        );
    }

    return (
        <aside className="w-96 h-full border-l border-white/10 bg-zinc-950/90 backdrop-blur-xl flex flex-col z-30 shadow-2xl shadow-black/50 animate-in slide-in-from-right duration-300">
            {/* Header */}
            <div className="p-4 border-b border-white/10 flex items-center justify-between">
                <div>
                    <h3 className="font-mono text-sm font-medium text-zinc-100">
                        IMG_{selectedImage.id.toString().padStart(4, '0')}
                    </h3>
                    <p className="text-xs text-zinc-500">Analysis & Metadata</p>
                </div>
                <Button variant="secondary" size="icon" onClick={onClose} className="h-8 w-8 hover:bg-white/10 border-transparent">
                    <X className="w-4 h-4" />
                </Button>
            </div>

            {/* Tabs */}
            <div className="flex border-b border-white/10 bg-black/20">
                {tabs.map((tab) => {
                    const Icon = tab.icon;
                    const isActive = activeTab === tab.id;
                    return (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={cn(
                                'flex-1 flex items-center justify-center gap-2 px-3 py-3 text-sm transition-all relative',
                                isActive
                                    ? 'text-cyan-400 bg-white/5'
                                    : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/5'
                            )}
                        >
                            <Icon className={cn("w-4 h-4", isActive && "text-cyan-400")} />
                            <span className="hidden lg:inline font-medium">{tab.label}</span>
                            {isActive && (
                                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-cyan-500 shadow-[0_0_8px_rgba(6,182,212,0.6)]" />
                            )}
                        </button>
                    );
                })}
            </div>

            {/* Tab Content */}
            <div className="flex-1 overflow-y-auto p-5 space-y-6">
                {activeTab === 'analysis' && <AnalysisTab image={selectedImage} />}
                {activeTab === 'captions' && <CaptionsTab />}
                {activeTab === 'settings' && <SettingsTab />}
            </div>
        </aside>
    );
}

function AnalysisTab({ image }) {
    return (
        <div className="space-y-6">
            {/* Score Overview */}
            <div className="p-5 rounded-xl bg-gradient-to-br from-zinc-900 to-black border border-white/10 relative overflow-hidden group">
                <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 group-hover:bg-cyan-500/20 transition-colors" />

                <div className="flex items-center justify-between mb-4 relative">
                    <span className="text-sm font-medium text-zinc-400">Overall Score</span>
                    <span className="text-3xl font-bold font-mono text-white tracking-tighter">
                        {(image.score * 100).toFixed(1)}%
                    </span>
                </div>
                <div className="h-2 bg-zinc-800 rounded-full overflow-hidden relative">
                    <div
                        className="h-full bg-gradient-to-r from-cyan-500 to-fuchsia-500 rounded-full transition-all shadow-[0_0_10px_rgba(6,182,212,0.4)]"
                        style={{ width: `${image.score * 100}%` }}
                    />
                </div>
            </div>

            {/* Metrics */}
            <div>
                <h4 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-4 pl-1">
                    Face Metrics
                </h4>
                <div className="space-y-4">
                    <MetricRow label="Face Similarity" value={0.87} />
                    <MetricRow label="Eye Distance" value={0.92} />
                    <MetricRow label="Face Angle" value={0.78} />
                </div>
            </div>

            <div>
                <h4 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-4 pl-1">
                    Body Metrics
                </h4>
                <div className="space-y-4">
                    <MetricRow label="Body Consistency" value={0.84} />
                    <MetricRow label="Pose Quality" value={0.91} />
                    <MetricRow label="Limb Ratios" value={0.88} />
                </div>
            </div>

            {/* Status */}
            <div className="p-4 rounded-xl bg-zinc-900/50 border border-white/10">
                <div className="flex items-center justify-between">
                    <span className="text-sm text-zinc-400">Status</span>
                    <span
                        className={cn(
                            'px-2.5 py-1 rounded-full text-xs font-medium capitalize flex items-center gap-1.5',
                            image.status === 'approved' && 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20',
                            image.status === 'rejected' && 'bg-red-500/10 text-red-400 border border-red-500/20',
                            image.status === 'analyzed' && 'bg-blue-500/10 text-blue-400 border border-blue-500/20',
                            image.status === 'pending' && 'bg-yellow-500/10 text-yellow-400 border border-yellow-500/20'
                        )}
                    >
                        <span className={cn("w-1.5 h-1.5 rounded-full inline-block",
                            image.status === 'approved' && "bg-emerald-400 shadow-[0_0_5px_rgba(52,211,153,0.5)]",
                            image.status === 'rejected' && "bg-red-400 shadow-[0_0_5px_rgba(248,113,113,0.5)]",
                            image.status === 'analyzed' && "bg-blue-400 shadow-[0_0_5px_rgba(96,165,250,0.5)]",
                            image.status === 'pending' && "bg-yellow-400 shadow-[0_0_5px_rgba(250,204,21,0.5)]"
                        )} />
                        {image.status}
                    </span>
                </div>
            </div>
        </div>
    );
}

function MetricRow({ label, value }) {
    // Dynamic color logic
    const color = value >= 0.85 ? 'bg-cyan-400' : value >= 0.70 ? 'bg-indigo-400' : 'bg-red-400';
    const glow = value >= 0.85 ? 'shadow-[0_0_8px_rgba(34,211,238,0.4)]' : '';

    return (
        <div>
            <div className="flex items-center justify-between text-xs mb-1.5">
                <span className="text-zinc-400">{label}</span>
                <span className="text-zinc-200 font-mono">{(value * 100).toFixed(0)}%</span>
            </div>
            <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                <div
                    className={cn('h-full rounded-full transition-all', color, glow)}
                    style={{ width: `${value * 100}%` }}
                />
            </div>
        </div>
    );
}

function CaptionsTab() {
    return (
        <div className="space-y-4">
            <CaptionBlock title="SDXL Caption" color="text-indigo-400" content="A portrait of a person with detailed features, professional lighting, high quality, 8k resolution" />
            <CaptionBlock title="Flux Caption" color="text-fuchsia-400" content="Portrait photograph, subject facing camera, natural expression, studio backdrop, soft diffused lighting" />
            <CaptionBlock title="Dense Caption" color="text-emerald-400" content="A detailed photograph showing a person in a portrait composition. The subject has distinctive facial features with well-defined bone structure." />
        </div>
    );
}

function CaptionBlock({ title, color, content }) {
    return (
        <div className="p-4 rounded-xl bg-zinc-900/30 border border-white/5 hover:border-white/10 transition-colors group">
            <div className="flex items-center justify-between mb-3">
                <span className={cn("text-xs font-semibold uppercase tracking-wider", color)}>{title}</span>
                <button className="text-[10px] bg-white/5 hover:bg-white/10 text-zinc-400 px-2 py-1 rounded transition-colors uppercase">
                    Regenerate
                </button>
            </div>
            <p className="text-sm text-zinc-400 leading-relaxed font-mono">
                {content}
            </p>
        </div>
    );
}

function SettingsTab() {
    return (
        <div className="space-y-8">
            <div>
                <div className="flex items-center gap-2 mb-4">
                    <Sliders className="w-4 h-4 text-zinc-400" />
                    <h4 className="text-sm font-medium text-zinc-200">
                        Threshold Settings
                    </h4>
                </div>
                <div className="space-y-6">
                    <SliderSetting label="Face Similarity Threshold" value={0.8} />
                    <SliderSetting label="Body Consistency Threshold" value={0.7} />
                    <SliderSetting label="Auto-reject Below" value={0.5} />
                </div>
            </div>

            <div className="border-t border-white/10 pt-6">
                <h4 className="text-sm font-medium text-zinc-200 mb-4">
                    Actions
                </h4>
                <div className="space-y-3">
                    <Button variant="primary" className="w-full bg-emerald-500/10 text-emerald-400 border-emerald-500/20 hover:bg-emerald-500/20 hover:shadow-[0_0_15px_rgba(16,185,129,0.2)]">
                        Approve Image
                    </Button>
                    <Button variant="danger" className="w-full">
                        Reject Image
                    </Button>
                </div>
            </div>
        </div>
    );
}

function SliderSetting({ label, value }) {
    return (
        <div>
            <div className="flex items-center justify-between text-xs mb-2">
                <span className="text-zinc-400">{label}</span>
                <span className="text-zinc-200 font-mono">{(value * 100).toFixed(0)}%</span>
            </div>
            <input
                type="range"
                min="0"
                max="100"
                value={value * 100}
                className="w-full h-1 bg-zinc-800 rounded-full appearance-none cursor-pointer
          [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3
          [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-cyan-500
          [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(6,182,212,0.5)]"
                readOnly
            />
        </div>
    );
}
