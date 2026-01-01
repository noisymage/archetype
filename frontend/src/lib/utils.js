import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Merge Tailwind classes with proper conflict resolution
 */
export function cn(...inputs) {
    return twMerge(clsx(inputs));
}
