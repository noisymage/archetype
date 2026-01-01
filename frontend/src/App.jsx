import { useEffect } from 'react';
import { Toaster, toast } from 'sonner';
import { ProjectProvider, useProject } from './context/ProjectContext';
import { Sidebar } from './components/Sidebar';
import { MainContent } from './components/MainContent';
import { DetailsPanel } from './components/DetailsPanel';
import { StatusIndicator } from './components/StatusIndicator';
import './App.css';

function AppContent() {
  const { selectedCharacter } = useProject();

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-black text-white font-sans selection:bg-cyan-500/30">

      {/* Toast Notifications */}
      <Toaster
        position="bottom-right"
        theme="dark"
        loadingIcon={null}
        toastOptions={{
          style: {
            background: 'rgba(9, 9, 11, 0.9)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            color: '#fff',
            backdropFilter: 'blur(8px)',
          }
        }}
      />

      {/* Main Layout */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Left Sidebar */}
        <Sidebar />

        {/* Main Content Area */}
        <MainContent />

        {/* Right Details Panel - hidden when no character selected */}
        {selectedCharacter && (
          <DetailsPanel />
        )}
      </div>

      {/* Footer */}
      <footer className="px-4 py-2 bg-zinc-950 border-t border-white/5 flex items-center justify-between z-20">
        <span className="text-xs text-zinc-500 font-mono">
          Archetype v0.2.0 • Character Consistency Validator
        </span>
        <StatusIndicator />
      </footer>
    </div>
  );
}

function App() {
  return (
    <ProjectProvider>
      <AppContent />
    </ProjectProvider>
  );
}

export default App;
