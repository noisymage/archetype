import { useState } from 'react';
import { BarChart3, MessageSquare, Settings, X, Sparkles, Sliders } from 'lucide-react';
import { cn } from '../lib/utils';

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
            <aside className="w-80 h-full bg-[var(--color-bg-secondary)] border-l border-[var(--color-border)] flex flex-col items-center justify-center p-6 text-center">
                <div className="w-16 h-16 rounded-2xl bg-[var(--color-bg-tertiary)] flex items-center justify-center mb-4">
                    <Sparkles className="w-8 h-8 text-[var(--color-text-muted)]" />
                </div>
                <h3 className="text-base font-medium text-[var(--color-text-primary)] mb-2">
                    Select an Image
                </h3>
                <p className="text-sm text-[var(--color-text-secondary)]">
                    Click on an image to view its analysis, captions, and settings.
                </p>
            </aside>
        );
    }

    return (
        <aside className="w-80 h-full bg-[var(--color-bg-secondary)] border-l border-[var(--color-border)] flex flex-col">
            {/* Header */}
            <div className="p-4 border-b border-[var(--color-border)] flex items-center justify-between">
                <h3 className="font-medium text-[var(--color-text-primary)]">
                    Image #{selectedImage.id}
                </h3>
                <button
                    onClick={onClose}
                    className="p-1.5 rounded-lg hover:bg-[var(--color-bg-tertiary)] transition-colors"
                >
                    <X className="w-4 h-4 text-[var(--color-text-secondary)]" />
                </button>
            </div>

            {/* Tabs */}
            <div className="flex border-b border-[var(--color-border)]">
                {tabs.map((tab) => {
                    const Icon = tab.icon;
                    return (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={cn(
                                'flex-1 flex items-center justify-center gap-2 px-3 py-3 text-sm transition-colors relative',
                                activeTab === tab.id
                                    ? 'text-[var(--color-accent)]'
                                    : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]'
                            )}
                        >
                            <Icon className="w-4 h-4" />
                            <span className="hidden lg:inline">{tab.label}</span>
                            {activeTab === tab.id && (
                                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-[var(--color-accent)]" />
                            )}
                        </button>
                    );
                })}
            </div>

            {/* Tab Content */}
            <div className="flex-1 overflow-y-auto p-4">
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
            <div className="p-4 rounded-xl bg-[var(--color-bg-tertiary)] border border-[var(--color-border)]">
                <div className="flex items-center justify-between mb-3">
                    <span className="text-sm text-[var(--color-text-secondary)]">Overall Score</span>
                    <span className="text-2xl font-semibold text-[var(--color-text-primary)]">
                        {(image.score * 100).toFixed(1)}%
                    </span>
                </div>
                <div className="h-2 bg-[var(--color-bg-elevated)] rounded-full overflow-hidden">
                    <div
                        className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all"
                        style={{ width: `${image.score * 100}%` }}
                    />
                </div>
            </div>

            {/* Metrics */}
            <div>
                <h4 className="text-sm font-medium text-[var(--color-text-primary)] mb-3">
                    Face Metrics
                </h4>
                <div className="space-y-3">
                    <MetricRow label="Face Similarity" value={0.87} />
                    <MetricRow label="Eye Distance" value={0.92} />
                    <MetricRow label="Face Angle" value={0.78} />
                </div>
            </div>

            <div>
                <h4 className="text-sm font-medium text-[var(--color-text-primary)] mb-3">
                    Body Metrics
                </h4>
                <div className="space-y-3">
                    <MetricRow label="Body Consistency" value={0.84} />
                    <MetricRow label="Pose Quality" value={0.91} />
                    <MetricRow label="Limb Ratios" value={0.88} />
                </div>
            </div>

            {/* Status */}
            <div className="p-4 rounded-xl bg-[var(--color-bg-tertiary)] border border-[var(--color-border)]">
                <div className="flex items-center justify-between">
                    <span className="text-sm text-[var(--color-text-secondary)]">Status</span>
                    <span
                        className={cn(
                            'px-2.5 py-1 rounded-full text-xs font-medium capitalize',
                            image.status === 'approved' && 'bg-emerald-500/10 text-emerald-500',
                            image.status === 'rejected' && 'bg-red-500/10 text-red-500',
                            image.status === 'analyzed' && 'bg-blue-500/10 text-blue-500',
                            image.status === 'pending' && 'bg-yellow-500/10 text-yellow-500'
                        )}
                    >
                        {image.status}
                    </span>
                </div>
            </div>
        </div>
    );
}

function MetricRow({ label, value }) {
    const color =
        value >= 0.85
            ? 'from-emerald-500 to-emerald-400'
            : value >= 0.7
                ? 'from-yellow-500 to-yellow-400'
                : 'from-red-500 to-red-400';

    return (
        <div>
            <div className="flex items-center justify-between text-xs mb-1.5">
                <span className="text-[var(--color-text-secondary)]">{label}</span>
                <span className="text-[var(--color-text-primary)]">{(value * 100).toFixed(0)}%</span>
            </div>
            <div className="h-1.5 bg-[var(--color-bg-elevated)] rounded-full overflow-hidden">
                <div
                    className={cn('h-full bg-gradient-to-r rounded-full', color)}
                    style={{ width: `${value * 100}%` }}
                />
            </div>
        </div>
    );
}

function CaptionsTab() {
    return (
        <div className="space-y-4">
            <div className="p-4 rounded-xl bg-[var(--color-bg-tertiary)] border border-[var(--color-border)]">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium text-indigo-400">SDXL Caption</span>
                    <button className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)]">
                        Regenerate
                    </button>
                </div>
                <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">
                    A portrait of a person with detailed features, professional lighting, high quality, 8k resolution
                </p>
            </div>

            <div className="p-4 rounded-xl bg-[var(--color-bg-tertiary)] border border-[var(--color-border)]">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium text-purple-400">Flux Caption</span>
                    <button className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)]">
                        Regenerate
                    </button>
                </div>
                <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">
                    Portrait photograph, subject facing camera, natural expression, studio backdrop, soft diffused lighting
                </p>
            </div>

            <div className="p-4 rounded-xl bg-[var(--color-bg-tertiary)] border border-[var(--color-border)]">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium text-emerald-400">Dense Caption</span>
                    <button className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)]">
                        Regenerate
                    </button>
                </div>
                <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">
                    A detailed photograph showing a person in a portrait composition. The subject has distinctive facial features with
                    well-defined bone structure. Professional studio lighting creates soft shadows and highlights...
                </p>
            </div>
        </div>
    );
}

function SettingsTab() {
    return (
        <div className="space-y-6">
            <div>
                <div className="flex items-center gap-2 mb-4">
                    <Sliders className="w-4 h-4 text-[var(--color-text-secondary)]" />
                    <h4 className="text-sm font-medium text-[var(--color-text-primary)]">
                        Threshold Settings
                    </h4>
                </div>
                <div className="space-y-4">
                    <SliderSetting label="Face Similarity Threshold" value={0.8} />
                    <SliderSetting label="Body Consistency Threshold" value={0.7} />
                    <SliderSetting label="Auto-reject Below" value={0.5} />
                </div>
            </div>

            <div className="border-t border-[var(--color-border)] pt-6">
                <h4 className="text-sm font-medium text-[var(--color-text-primary)] mb-4">
                    Actions
                </h4>
                <div className="space-y-2">
                    <button className="w-full px-4 py-2.5 rounded-lg bg-emerald-500/10 text-emerald-500 text-sm font-medium hover:bg-emerald-500/20 transition-colors">
                        Approve Image
                    </button>
                    <button className="w-full px-4 py-2.5 rounded-lg bg-red-500/10 text-red-500 text-sm font-medium hover:bg-red-500/20 transition-colors">
                        Reject Image
                    </button>
                </div>
            </div>
        </div>
    );
}

function SliderSetting({ label, value }) {
    return (
        <div>
            <div className="flex items-center justify-between text-xs mb-2">
                <span className="text-[var(--color-text-secondary)]">{label}</span>
                <span className="text-[var(--color-text-primary)]">{(value * 100).toFixed(0)}%</span>
            </div>
            <input
                type="range"
                min="0"
                max="100"
                value={value * 100}
                className="w-full h-1.5 bg-[var(--color-bg-elevated)] rounded-full appearance-none cursor-pointer
          [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4
          [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[var(--color-accent)]
          [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:shadow-lg"
                readOnly
            />
        </div>
    );
}
