// Automatic environment detection
const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

const LOCAL_API = "http://localhost:8000/api";
const PROD_API = "https://aegis-api-762161152188.asia-south1.run.app/api";

export const API_URL = isLocal ? LOCAL_API : PROD_API;
