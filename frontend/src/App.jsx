import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Players from './pages/Players';
import GraphView from './pages/GraphView';
import Analytics from './pages/Analytics';
import Sandbox from './pages/Sandbox';
import './App.css';

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-layout">
        <div className="animated-background" aria-hidden="true">
          <div className="background-grid" />
          <div className="background-sweep" />
          <div className="background-streams" />
        </div>
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/players" element={<Players />} />
            <Route path="/graph" element={<GraphView />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/sandbox" element={<Sandbox />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
