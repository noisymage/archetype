import { useEffect, useState } from 'react';
import { Wifi, WifiOff } from 'lucide-react';

/**
 * Status indicator component that checks backend health
 */
export function StatusIndicator() {
    const [status, setStatus] = useState('checking');

    useEffect(() => {
        const checkHealth = async () => {
            try {
                const response = await fetch('/api/health');
                if (response.ok) {
                    const data = await response.json();
                    if (data.status === 'ok') {
                        setStatus('connected');
                    } else {
                        setStatus('error');
                    }
                } else {
                    setStatus('error');
                }
            } catch (error) {
                setStatus('disconnected');
            }
        };

        checkHealth();
        // Check every 30 seconds
        const interval = setInterval(checkHealth, 30000);
        return () => clearInterval(interval);
    }, []);

    const statusConfig = {
        checking: {
            color: 'bg-yellow-500',
            text: 'Checking...',
            icon: Wifi,
        },
        connected: {
            color: 'bg-emerald-500',
            text: 'Connected',
            icon: Wifi,
        },
        disconnected: {
            color: 'bg-red-500',
            text: 'Disconnected',
            icon: WifiOff,
        },
        error: {
            color: 'bg-red-500',
            text: 'Error',
            icon: WifiOff,
        },
    };

    const config = statusConfig[status];
    const Icon = config.icon;

    return (
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[var(--color-bg-tertiary)] border border-[var(--color-border)]">
            <span className={`w-2 h-2 rounded-full ${config.color} animate-pulse`} />
            <Icon className="w-3.5 h-3.5 text-[var(--color-text-secondary)]" />
            <span className="text-xs text-[var(--color-text-secondary)]">{config.text}</span>
        </div>
    );
}
