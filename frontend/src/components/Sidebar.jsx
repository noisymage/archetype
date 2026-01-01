import { useState } from 'react';
import { FolderOpen, User, ChevronRight, Plus, Settings } from 'lucide-react';
import { cn } from '../lib/utils';
import { Button } from './ui/Button';

// Mock data for projects and characters
const mockProjects = [
    {
        id: 1,
        name: 'Fantasy Portrait Set',
        characters: [
            { id: 1, name: 'Elara the Mage', imageCount: 45 },
            { id: 2, name: 'Thorin Warrior', imageCount: 32 },
        ],
    },
    {
        id: 2,
        name: 'Sci-Fi Characters',
        characters: [
            { id: 3, name: 'Nova Pilot', imageCount: 28 },
            { id: 4, name: 'Android X-7', imageCount: 51 },
        ],
    },
    {
        id: 3,
        name: 'Portrait Training',
        characters: [
            { id: 5, name: 'Reference Model A', imageCount: 120 },
        ],
    },
];

/**
 * Left sidebar for project/character navigation
 */
export function Sidebar({ selectedCharacter, onSelectCharacter }) {
    const [expandedProjects, setExpandedProjects] = useState([1]);

    const toggleProject = (projectId) => {
        setExpandedProjects((prev) =>
            prev.includes(projectId)
                ? prev.filter((id) => id !== projectId)
                : [...prev, projectId]
        );
    };

    return (
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
                    <button className="p-1 rounded hover:bg-white/5 transition-colors text-zinc-500 hover:text-white">
                        <Plus className="w-3 h-3" />
                    </button>
                </div>

                <div className="space-y-1">
                    {mockProjects.map((project) => (
                        <div key={project.id}>
                            <button
                                onClick={() => toggleProject(project.id)}
                                className="w-full flex items-center gap-2 px-2 py-2 rounded-lg hover:bg-white/5 transition-colors group text-zinc-300 hover:text-white"
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
                            </button>

                            {expandedProjects.includes(project.id) && (
                                <div className="ml-2 pl-3 border-l border-white/5 mt-1 space-y-0.5">
                                    {project.characters.map((character) => {
                                        const isSelected = selectedCharacter?.id === character.id;
                                        return (
                                            <button
                                                key={character.id}
                                                onClick={() => onSelectCharacter(character)}
                                                className={cn(
                                                    'w-full flex items-center gap-2 px-3 py-2 rounded-md transition-all relative group/item',
                                                    isSelected
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
                                                    {character.imageCount}
                                                </span>
                                            </button>
                                        );
                                    })}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* Footer */}
            <div className="p-3 border-t border-white/10 bg-black/20 backdrop-blur-xl">
                <Button variant="primary" className="w-full gap-2">
                    <Plus className="w-4 h-4" />
                    New Project
                </Button>
            </div>
        </aside>
    );
}
