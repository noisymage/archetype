import { useState } from 'react';
import { FolderOpen, User, ChevronRight, Plus, Settings, Trash2, Loader2 } from 'lucide-react';
import { cn } from '../lib/utils';
import { Button } from './ui/Button';
import { useProject } from '../context/ProjectContext';
import { CreateProjectWizard } from './CreateProjectWizard';

/**
 * Left sidebar for project/character navigation
 */
export function Sidebar({ currentView, onViewChange }) {
    const {
        projects,
        selectedCharacter,
        selectCharacter,
        deleteProject,
        deleteCharacter
    } = useProject();

    const [expandedProjects, setExpandedProjects] = useState([]);
    const [showWizard, setShowWizard] = useState(false);
    const [deletingId, setDeletingId] = useState(null);

    const [projectForWizard, setProjectForWizard] = useState(null);

    const toggleProject = (projectId) => {
        setExpandedProjects((prev) =>
            prev.includes(projectId)
                ? prev.filter((id) => id !== projectId)
                : [...prev, projectId]
        );
    };

    const handleAddCharacter = (e, project) => {
        e.stopPropagation();
        setProjectForWizard(project);
        setShowWizard(true);
    };

    const handleCloseWizard = () => {
        setShowWizard(false);
        setProjectForWizard(null);
    };

    const handleDeleteProject = async (e, projectId) => {
        e.stopPropagation();
        if (confirm('Delete this project and all its characters?')) {
            setDeletingId(projectId);
            try {
                await deleteProject(projectId);
            } finally {
                setDeletingId(null);
            }
        }
    };

    const handleDeleteCharacter = async (e, characterId) => {
        e.stopPropagation();
        if (confirm('Delete this character and all its data?')) {
            setDeletingId(characterId);
            try {
                await deleteCharacter(characterId);
            } finally {
                setDeletingId(null);
            }
        }
    };

    return (
        <>
            <aside className="w-64 h-full bg-zinc-950/80 backdrop-blur-md border-r border-white/10 flex flex-col relative z-20">
                {/* Header */}
                <div className="p-4 border-b border-white/10">
                    <div className="flex items-center justify-between">
                        <h1 className="text-lg font-semibold bg-gradient-to-r from-cyan-400 to-fuchsia-500 bg-clip-text text-transparent tracking-tight">
                            Archetype
                        </h1>
                        <button className="p-1.5 rounded-lg hover:bg-white/5 transition-colors text-zinc-400 hover:text-white">
                            <Settings className="w-4 h-4" />
                        </button>
                    </div>
                </div>

                {/* Project List */}
                <div className="flex-1 overflow-y-auto p-3 space-y-4">
                    <div className="flex items-center justify-between px-2">
                        <span className="text-xs font-mono font-medium text-zinc-500 uppercase tracking-wider">
                            Projects
                        </span>
                        <span className="text-xs text-zinc-600">{projects.length}</span>
                    </div>

                    {projects.length === 0 ? (
                        <div className="px-2 py-8 text-center">
                            <p className="text-sm text-zinc-500 mb-4">No projects yet</p>
                            <Button
                                variant="secondary"
                                size="sm"
                                onClick={() => setShowWizard(true)}
                                className="gap-2"
                            >
                                <Plus className="w-4 h-4" />
                                Create First Project
                            </Button>
                        </div>
                    ) : (
                        <div className="space-y-1">
                            {projects.map((project) => (
                                <div key={project.id}>
                                    <div
                                        onClick={() => toggleProject(project.id)}
                                        className="w-full flex items-center gap-2 px-2 py-2 rounded-lg hover:bg-white/5 transition-colors group text-zinc-300 hover:text-white cursor-pointer select-none"
                                    >
                                        <ChevronRight
                                            className={cn(
                                                'w-4 h-4 text-zinc-600 transition-transform duration-200 group-hover:text-zinc-400',
                                                expandedProjects.includes(project.id) && 'rotate-90'
                                            )}
                                        />
                                        <FolderOpen className="w-4 h-4 text-cyan-500/70 group-hover:text-cyan-400 transition-colors" />
                                        <span className="text-sm font-medium truncate flex-1 text-left">
                                            {project.name}
                                        </span>

                                        {/* Add Character Button */}
                                        <button
                                            onClick={(e) => handleAddCharacter(e, project)}
                                            className="opacity-0 group-hover:opacity-100 p-1 hover:bg-white/10 rounded transition-all mr-1"
                                            title="Add Character"
                                        >
                                            <Plus className="w-3 h-3 text-zinc-400 hover:text-white" />
                                        </button>

                                        {/* Delete Project Button */}
                                        <button
                                            onClick={(e) => handleDeleteProject(e, project.id)}
                                            className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-500/20 rounded transition-all"
                                            disabled={deletingId === project.id}
                                        >
                                            {deletingId === project.id ? (
                                                <Loader2 className="w-3 h-3 animate-spin text-zinc-400" />
                                            ) : (
                                                <Trash2 className="w-3 h-3 text-red-400" />
                                            )}
                                        </button>
                                    </div>

                                    {expandedProjects.includes(project.id) && (
                                        <div className="ml-2 pl-3 border-l border-white/5 mt-1 space-y-0.5">
                                            {(project.characters || []).map((character) => {
                                                const isSelected = selectedCharacter?.id === character.id;
                                                return (
                                                    <button
                                                        key={character.id}
                                                        onClick={() => {
                                                            selectCharacter(character);
                                                            onViewChange?.('dashboard');
                                                        }}
                                                        className={cn(
                                                            'w-full flex items-center gap-2 px-3 py-2 rounded-md transition-all relative group/item',
                                                            isSelected && currentView === 'dashboard'
                                                                ? 'bg-cyan-500/10 text-cyan-400'
                                                                : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/5'
                                                        )}
                                                    >
                                                        {/* Active Indicator Glow */}
                                                        {isSelected && (
                                                            <div className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-4 bg-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.5)] rounded-r-full" />
                                                        )}

                                                        <User className={cn("w-3.5 h-3.5", isSelected ? "text-cyan-400" : "text-zinc-600 group-hover/item:text-zinc-400")} />
                                                        <span className="text-sm truncate flex-1 text-left">
                                                            {character.name}
                                                        </span>
                                                        <span className="text-xs font-mono text-zinc-600 group-hover/item:text-zinc-500">
                                                            {character.image_count || 0}
                                                        </span>
                                                        <button
                                                            onClick={(e) => handleDeleteCharacter(e, character.id)}
                                                            className="opacity-0 group-hover/item:opacity-100 p-1 hover:bg-red-500/20 rounded transition-all"
                                                            disabled={deletingId === character.id}
                                                        >
                                                            {deletingId === character.id ? (
                                                                <Loader2 className="w-3 h-3 animate-spin text-zinc-400" />
                                                            ) : (
                                                                <Trash2 className="w-3 h-3 text-red-400" />
                                                            )}
                                                        </button>
                                                    </button>
                                                );
                                            })}
                                            {(project.characters || []).length === 0 && (
                                                <p className="text-xs text-zinc-600 px-3 py-2">No characters</p>
                                            )}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Tools Section */}
                <div className="px-3 pb-3">
                    <div className="flex items-center justify-between px-2 mb-2 mt-4">
                        <span className="text-xs font-mono font-medium text-zinc-500 uppercase tracking-wider">
                            Tools
                        </span>
                    </div>
                    <button
                        onClick={() => onViewChange?.('tool-compare')}
                        className={cn(
                            'w-full flex items-center gap-2 px-3 py-2 rounded-md transition-all group',
                            currentView === 'tool-compare'
                                ? 'bg-cyan-500/10 text-cyan-400'
                                : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/5'
                        )}
                    >
                        <User className="w-4 h-4" />
                        <span className="text-sm font-medium">Compare Images</span>
                    </button>
                </div>

                {/* Footer */}
                <div className="p-3 border-t border-white/10 bg-black/20 backdrop-blur-xl">
                    <Button
                        variant="primary"
                        className="w-full gap-2"
                        onClick={() => setShowWizard(true)}
                    >
                        <Plus className="w-4 h-4" />
                        New Project
                    </Button>
                </div>
            </aside>

            {/* Create Project Wizard - Conditionally rendered to ensure fresh state */}
            {showWizard && (
                <CreateProjectWizard
                    isOpen={true}
                    onClose={handleCloseWizard}
                    initialProject={projectForWizard}
                />
            )}
        </>
    );
}
