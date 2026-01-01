import { useState, useEffect } from 'react';
import { Toaster, toast } from 'sonner';
import { Sidebar } from './components/Sidebar';
import { MainContent } from './components/MainContent';
import { DetailsPanel } from './components/DetailsPanel';
import { StatusIndicator } from './components/StatusIndicator';
import './App.css';

function App() {
  const [selectedCharacter, setSelectedCharacter] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // Simulate loading state when selecting a character
  useEffect(() => {
    if (selectedCharacter) {
      setIsLoading(true);
      toast.info('Loading character dataset...', { duration: 1500 });
      const timer = setTimeout(() => {
        setIsLoading(false);
        toast.success(`Loaded ${selectedCharacter.name} successfully`);
      }, 1500);
      return () => clearTimeout(timer);
    }
  }, [selectedCharacter]);

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
        <Sidebar
          selectedCharacter={selectedCharacter}
          onSelectCharacter={(char) => {
            if (selectedCharacter?.id !== char.id) {
              setSelectedCharacter(char);
              setSelectedImage(null);
            }
          }}
        />

        {/* Main Content Area */}
        <MainContent
          selectedCharacter={selectedCharacter}
          selectedImage={selectedImage}
          onSelectImage={setSelectedImage}
          isLoading={isLoading}
        />

        {/* Right Details Panel */}
        <DetailsPanel
          selectedImage={selectedImage}
          onClose={() => setSelectedImage(null)}
        />
      </div>

      {/* Footer */}
      <footer className="px-4 py-2 bg-zinc-950 border-t border-white/5 flex items-center justify-between z-20">
        <span className="text-xs text-zinc-500 font-mono">
          Archetype v0.1.0 • Character Consistency Validator
        </span>
        <StatusIndicator />
      </footer>
    </div>
  );
}

export default App;
