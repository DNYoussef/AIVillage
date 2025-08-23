import React from 'react';
import ReactDOM from 'react-dom/client';
import App from '../App';
import '../App.css';

// Enhanced main entry point for consolidated UI
const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);