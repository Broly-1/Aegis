export const getApiUrl = () => {
    const saved = localStorage.getItem('api_env');
    if (saved === 'local') return 'http://localhost:8000/api';
    if (saved === 'remote') return 'https://aegis-api-762161152188.us-central1.run.app/api';
    
    // Default to the .env variable, or fallback to localhost
    return import.meta.env.VITE_API_URL || 'http://localhost:8000/api';
};

export const setApiEnv = (env) => {
    localStorage.setItem('api_env', env);
    window.location.reload(); // Reload to apply the new URL across all components
};

export const getApiEnv = () => {
    const saved = localStorage.getItem('api_env');
    if (saved) return saved;
    // Guess default based on URL
    return import.meta.env.VITE_API_URL?.includes('localhost') ? 'local' : 'remote';
};
