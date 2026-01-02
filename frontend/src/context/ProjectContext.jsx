/**
 * Project Context - State management for projects, characters, and batch processing
 */
import { createContext, useContext, useState, useCallback, useEffect, useRef } from 'react';
import { toast } from 'sonner';
import * as api from '../lib/api';

const ProjectContext = createContext(null);

export function ProjectProvider({ children }) {
    // Projects and characters state
    const [projects, setProjects] = useState([]);
    const [selectedProject, setSelectedProject] = useState(null);
    const [selectedCharacter, setSelectedCharacter] = useState(null);
    const [datasetImages, setDatasetImages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);

    // Processing job state
    const [activeJob, setActiveJob] = useState(null);
    const wsRef = useRef(null);

    // Fetch projects on mount
    useEffect(() => {
        loadProjects();
    }, []);

    // Fetch characters when project changes
    useEffect(() => {
        if (selectedProject) {
            loadCharacters(selectedProject.id);
        }
    }, [selectedProject?.id]);

    // Fetch images when character changes
    useEffect(() => {
        if (selectedCharacter) {
            loadDatasetImages(selectedCharacter.id);
        } else {
            setDatasetImages([]);
        }
    }, [selectedCharacter?.id]);

    // Cleanup WebSocket on unmount
    useEffect(() => {
        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, []);

    // === Project Operations ===

    const loadProjects = useCallback(async () => {
        try {
            const data = await api.fetchProjects();
            setProjects(data);
        } catch (error) {
            console.error('Failed to load projects:', error);
            toast.error('Failed to load projects');
        }
    }, []);

    const createProject = useCallback(async (name, loraPresetType = 'SDXL') => {
        try {
            const project = await api.createProject(name, loraPresetType);
            setProjects(prev => [...prev, { ...project, characters: [] }]);
            toast.success(`Project "${name}" created`);
            return project;
        } catch (error) {
            toast.error(`Failed to create project: ${error.message}`);
            throw error;
        }
    }, []);

    const deleteProject = useCallback(async (projectId) => {
        try {
            await api.deleteProject(projectId);
            setProjects(prev => prev.filter(p => p.id !== projectId));
            if (selectedProject?.id === projectId) {
                setSelectedProject(null);
                setSelectedCharacter(null);
            }
            toast.success('Project deleted');
        } catch (error) {
            toast.error(`Failed to delete project: ${error.message}`);
            throw error;
        }
    }, [selectedProject]);

    // === Character Operations ===

    const loadCharacters = useCallback(async (projectId) => {
        try {
            const characters = await api.fetchCharacters(projectId);
            setProjects(prev => prev.map(p =>
                p.id === projectId ? { ...p, characters } : p
            ));
        } catch (error) {
            console.error('Failed to load characters:', error);
        }
    }, []);

    const createCharacter = useCallback(async (projectId, name, gender = 'neutral') => {
        try {
            const character = await api.createCharacter(projectId, name, gender);
            setProjects(prev => prev.map(p =>
                p.id === projectId
                    ? { ...p, characters: [...(p.characters || []), character] }
                    : p
            ));
            toast.success(`Character "${name}" created`);
            return character;
        } catch (error) {
            toast.error(`Failed to create character: ${error.message}`);
            throw error;
        }
    }, []);

    const deleteCharacter = useCallback(async (characterId) => {
        try {
            await api.deleteCharacter(characterId);
            setProjects(prev => prev.map(p => ({
                ...p,
                characters: (p.characters || []).filter(c => c.id !== characterId)
            })));
            if (selectedCharacter?.id === characterId) {
                setSelectedCharacter(null);
            }
            toast.success('Character deleted');
        } catch (error) {
            toast.error(`Failed to delete character: ${error.message}`);
            throw error;
        }
    }, [selectedCharacter]);

    const selectCharacter = useCallback((character) => {
        setSelectedCharacter(character);
        // Find and set the parent project
        const project = projects.find(p =>
            p.characters?.some(c => c.id === character.id)
        );
        if (project) {
            setSelectedProject(project);
        }
    }, [projects]);

    // === Dataset Operations ===

    const loadDatasetImages = useCallback(async (characterId) => {
        setIsLoading(true);
        try {
            const images = await api.fetchDatasetImages(characterId);
            setDatasetImages(images);
        } catch (error) {
            console.error('Failed to load images:', error);
            toast.error('Failed to load dataset images');
        } finally {
            setIsLoading(false);
        }
    }, []);

    const scanFolder = useCallback(async (folderPath, characterId) => {
        try {
            const result = await api.scanFolder(folderPath, characterId);
            toast.success(`Found ${result.total_found} images (${result.new_entries} new)`);
            // Reload images
            await loadDatasetImages(characterId);
            return result;
        } catch (error) {
            toast.error(`Scan failed: ${error.message}`);
            throw error;
        }
    }, [loadDatasetImages]);

    // === Batch Processing ===

    const startProcessing = useCallback(async (characterId, reprocessAll = false) => {
        try {
            const job = await api.startBatchProcess(characterId, reprocessAll);

            setActiveJob({
                ...job,
                processed: 0,
                currentImage: null
            });

            // Connect WebSocket for progress
            const { ws, close } = api.createProgressWebSocket(job.job_id, (data) => {
                if (data.type === 'progress') {
                    setActiveJob(prev => ({
                        ...prev,
                        processed: data.processed,
                        total: data.total,
                        currentImage: data.current_image
                    }));
                } else if (data.type === 'completed') {
                    toast.success('Processing completed!');
                    setActiveJob(null);
                    loadDatasetImages(characterId);
                    close();
                } else if (data.type === 'cancelled') {
                    toast.info('Processing cancelled');
                    setActiveJob(null);
                    loadDatasetImages(characterId);
                    close();
                } else if (data.type === 'error') {
                    toast.error(`Processing error: ${data.message}`);
                    setActiveJob(null);
                    close();
                }
            });

            wsRef.current = { close };
            toast.info(`Started processing ${job.total_images} images`);
            return job;
        } catch (error) {
            toast.error(`Failed to start processing: ${error.message}`);
            throw error;
        }
    }, [loadDatasetImages]);

    const cancelProcessing = useCallback(async () => {
        if (!activeJob) return;

        try {
            await api.cancelJob(activeJob.job_id);
            toast.info('Cancellation requested...');
        } catch (error) {
            toast.error(`Failed to cancel: ${error.message}`);
        }
    }, [activeJob]);

    // === Reference Images ===

    const setReferenceImages = useCallback(async (characterId, images, sourcePath = null) => {
        try {
            await api.setReferenceImages(characterId, images, sourcePath);
            toast.success('Reference images saved');
        } catch (error) {
            toast.error(`Failed to save references: ${error.message}`);
            throw error;
        }
    }, []);

    const analyzeReferences = useCallback(async (images, gender = 'neutral') => {
        try {
            return await api.analyzeReferences(images, gender);
        } catch (error) {
            toast.error(`Reference analysis failed: ${error.message}`);
            throw error;
        }
    }, []);

    const value = {
        // State
        projects,
        selectedProject,
        selectedCharacter,
        datasetImages,
        isLoading,
        activeJob,

        // Actions
        loadProjects,
        createProject,
        deleteProject,
        setSelectedProject,

        loadCharacters,
        createCharacter,
        deleteCharacter,
        selectCharacter,
        setSelectedCharacter,

        loadDatasetImages,
        scanFolder,

        startProcessing,
        cancelProcessing,

        setReferenceImages,
        analyzeReferences
    };

    return (
        <ProjectContext.Provider value={value}>
            {children}
        </ProjectContext.Provider>
    );
}

export function useProject() {
    const context = useContext(ProjectContext);
    if (!context) {
        throw new Error('useProject must be used within a ProjectProvider');
    }
    return context;
}
