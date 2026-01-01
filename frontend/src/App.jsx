import { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { MainContent } from './components/MainContent';
import { DetailsPanel } from './components/DetailsPanel';
import { StatusIndicator } from './components/StatusIndicator';
import './App.css';

function App() {
  const [selectedCharacter, setSelectedCharacter] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {/* Main Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar */}
        <Sidebar
          selectedCharacter={selectedCharacter}
          onSelectCharacter={setSelectedCharacter}
        />

        {/* Main Content Area */}
        <MainContent
          selectedCharacter={selectedCharacter}
          selectedImage={selectedImage}
          onSelectImage={setSelectedImage}
        />

        {/* Right Details Panel */}
        <DetailsPanel
          selectedImage={selectedImage}
          onClose={() => setSelectedImage(null)}
        />
      </div>

      {/* Footer */}
      <footer className="px-4 py-2 bg-[var(--color-bg-secondary)] border-t border-[var(--color-border)] flex items-center justify-between">
        <span className="text-xs text-[var(--color-text-muted)]">
          Archetype v0.1.0 • Character Consistency Validator
        </span>
        <StatusIndicator />
      </footer>
    </div>
  );
}

export default App;
