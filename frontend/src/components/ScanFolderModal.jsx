import { useState, useEffect } from 'react';
import { X, Search, Loader2, FolderOpen, AlertTriangle, CheckCircle } from 'lucide-react';
import { cn } from '../lib/utils';
import { Button } from './ui/Button';
import * as api from '../lib/api';
import { useProject } from '../context/ProjectContext';

/**
 * Modal to scan a folder for dataset images
 */
export function ScanFolderModal({ open, onClose, character }) {
    const { loadDatasetImages } = useProject();
    const [folderPath, setFolderPath] = useState('');
    const [isScanning, setIsScanning] = useState(false);
    const [scanResult, setScanResult] = useState(null);
    const [error, setError] = useState(null);

    // Auto-fill from existing images if available
    useEffect(() => {
        if (open && character?.datasetImages?.length > 0) {
            // Get directory of the first image
            const firstImg = character.datasetImages[0];
            if (firstImg.original_path) {
                const dir = firstImg.original_path.substring(0, firstImg.original_path.lastIndexOf('/'));
                setFolderPath(dir);
            }
        } else {
            setFolderPath('');
            setScanResult(null);
            setError(null);
        }
    }, [open, character]);

    const handleScan = async () => {
        if (!folderPath.trim()) return;

        setIsScanning(true);
        setError(null);
        setScanResult(null);

        try {
            const result = await api.scanFolder(folderPath, character.id);
            setScanResult(result);
            await loadDatasetImages(character.id); // Refresh images
        } catch (err) {
            setError(err.message || 'Failed to scan folder');
        } finally {
            setIsScanning(false);
        }
    };

    if (!open) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-[fadeIn_0.2s_ease-out]">
            <div
                className="bg-zinc-900 border border-white/10 rounded-xl shadow-2xl w-full max-w-lg overflow-hidden animate-[scaleIn_0.2s_ease-out]"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between bg-zinc-950/50">
                    <h3 className="text-lg font-semibold text-zinc-100 flex items-center gap-2">
                        <FolderOpen className="w-5 h-5 text-cyan-500" />
                        Scan Dataset Folder
                    </h3>
                    <button
                        onClick={onClose}
                        className="p-1 hover:bg-white/10 rounded-lg transition-colors text-zinc-400 hover:text-white"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Content */}
                <div className="p-6 space-y-6">
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-zinc-300">
                            Folder Path
                        </label>
                        <div className="flex gap-2">
                            <div className="relative flex-1">
                                <FolderOpen className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
                                <input
                                    type="text"
                                    value={folderPath}
                                    onChange={(e) => setFolderPath(e.target.value)}
                                    placeholder="/absolute/path/to/images"
                                    className="w-full bg-black/50 border border-white/10 rounded-lg py-2 pl-9 pr-3 text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/50 transition-all font-mono"
                                    onKeyDown={(e) => e.key === 'Enter' && handleScan()}
                                />
                            </div>
                        </div>
                        <p className="text-xs text-zinc-500">
                            Provide the absolute path to the folder containing your dataset images.
                        </p>
                    </div>

                    {error && (
                        <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-sm text-red-400 flex items-start gap-2">
                            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                            {error}
                        </div>
                    )}

                    {scanResult && (
                        <div className="p-4 bg-green-500/10 border border-green-500/20 rounded-lg space-y-2">
                            <div className="flex items-center gap-2 text-green-400 font-medium">
                                <CheckCircle className="w-4 h-4" />
                                Scan Complete
                            </div>
                            <div className="grid grid-cols-3 gap-2 text-xs text-zinc-400">
                                <div className="bg-black/20 p-2 rounded text-center">
                                    <div className="text-lg font-bold text-zinc-200">{scanResult.total_found}</div>
                                    <div>Found</div>
                                </div>
                                <div className="bg-black/20 p-2 rounded text-center">
                                    <div className="text-lg font-bold text-green-400">{scanResult.new_entries}</div>
                                    <div>New</div>
                                </div>
                                <div className="bg-black/20 p-2 rounded text-center">
                                    <div className="text-lg font-bold text-zinc-500">{scanResult.already_exists}</div>
                                    <div>Existing</div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="px-6 py-4 border-t border-white/5 bg-zinc-950/50 flex justify-end gap-2">
                    <Button variant="ghost" onClick={onClose}>
                        Close
                    </Button>
                    <Button
                        onClick={handleScan}
                        disabled={isScanning || !folderPath.trim()}
                        className="gap-2 min-w-[100px]"
                    >
                        {isScanning ? (
                            <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                Scanning...
                            </>
                        ) : (
                            <>
                                <Search className="w-4 h-4" />
                                {scanResult ? 'Rescan' : 'Scan Folder'}
                            </>
                        )}
                    </Button>
                </div>
            </div>
        </div>
    );
}
