import { useState } from 'react';
import { FolderOpen, User, ChevronRight, Plus, Settings } from 'lucide-react';
import { cn } from '../lib/utils';

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
        <aside className="w-64 h-full bg-[var(--color-bg-secondary)] border-r border-[var(--color-border)] flex flex-col">
            {/* Header */}
            <div className="p-4 border-b border-[var(--color-border)]">
                <div className="flex items-center justify-between">
                    <h1 className="text-lg font-semibold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
                        Archetype
                    </h1>
                    <button className="p-1.5 rounded-lg hover:bg-[var(--color-bg-tertiary)] transition-colors">
                        <Settings className="w-4 h-4 text-[var(--color-text-secondary)]" />
                    </button>
                </div>
            </div>

            {/* Project List */}
            <div className="flex-1 overflow-y-auto p-3">
                <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wider">
                        Projects
                    </span>
                    <button className="p-1 rounded hover:bg-[var(--color-bg-tertiary)] transition-colors">
                        <Plus className="w-4 h-4 text-[var(--color-text-secondary)]" />
                    </button>
                </div>

                <div className="space-y-1">
                    {mockProjects.map((project) => (
                        <div key={project.id}>
                            <button
                                onClick={() => toggleProject(project.id)}
                                className="w-full flex items-center gap-2 px-2 py-2 rounded-lg hover:bg-[var(--color-bg-tertiary)] transition-colors group"
                            >
                                <ChevronRight
                                    className={cn(
                                        'w-4 h-4 text-[var(--color-text-muted)] transition-transform',
                                        expandedProjects.includes(project.id) && 'rotate-90'
                                    )}
                                />
                                <FolderOpen className="w-4 h-4 text-indigo-400" />
                                <span className="text-sm text-[var(--color-text-primary)] truncate flex-1 text-left">
                                    {project.name}
                                </span>
                            </button>

                            {expandedProjects.includes(project.id) && (
                                <div className="ml-6 mt-1 space-y-0.5">
                                    {project.characters.map((character) => (
                                        <button
                                            key={character.id}
                                            onClick={() => onSelectCharacter(character)}
                                            className={cn(
                                                'w-full flex items-center gap-2 px-2 py-1.5 rounded-lg transition-colors',
                                                selectedCharacter?.id === character.id
                                                    ? 'bg-[var(--color-accent-muted)] text-[var(--color-accent)]'
                                                    : 'hover:bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)]'
                                            )}
                                        >
                                            <User className="w-3.5 h-3.5" />
                                            <span className="text-sm truncate flex-1 text-left">
                                                {character.name}
                                            </span>
                                            <span className="text-xs text-[var(--color-text-muted)]">
                                                {character.imageCount}
                                            </span>
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* Footer */}
            <div className="p-3 border-t border-[var(--color-border)]">
                <button className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-[var(--color-accent)] hover:bg-[var(--color-accent-hover)] transition-colors text-white text-sm font-medium">
                    <Plus className="w-4 h-4" />
                    New Project
                </button>
            </div>
        </aside>
    );
}
