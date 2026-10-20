import React, { useState, useEffect } from 'react';
import { AlertTriangle, FileQuestion } from 'lucide-react';

const globalDataCache = {};
const pendingPromises = {};

export function preloadData(filename) {
  const cleanName = filename.replace(/^\/?data\//, '');
  if (globalDataCache[cleanName]) return Promise.resolve(globalDataCache[cleanName]);
  if (pendingPromises[cleanName]) return pendingPromises[cleanName];

  const fullPath = `/data/${cleanName}`;
  const promise = fetch(fullPath)
    .then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      return res.json();
    })
    .then((json) => {
      globalDataCache[cleanName] = json;
      delete pendingPromises[cleanName];
      return json;
    })
    .catch((err) => {
      delete pendingPromises[cleanName];
      throw err;
    });

  pendingPromises[cleanName] = promise;
  return promise;
}

export function useData(filename) {
  const cleanName = filename ? filename.replace(/^\/?data\//, '') : '';
  const cached = globalDataCache[cleanName];
  const [data, setData] = useState(cached || null);
  const [loading, setLoading] = useState(!cached);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;

    if (globalDataCache[cleanName]) {
      setData(globalDataCache[cleanName]);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    preloadData(cleanName)
      .then((json) => {
        if (isMounted) {
          setData(json);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message || 'File not found');
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [cleanName]);

  const missing = !loading && (error !== null || data === null);
  return { 
    data, 
    loading, 
    error, 
    path: `/data/${cleanName}`, 
    missing, 
    missingPath: `/data/${cleanName}` 
  };
}

export function DataMissingPanel({ path, error }) {
  return React.createElement(
    'div',
    { className: 'rounded-2xl border border-rose-500/30 bg-rose-950/20 p-8 text-center backdrop-blur-md' },
    React.createElement(
      'div',
      { className: 'mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-rose-500/20 text-rose-400' },
      React.createElement(FileQuestion, { className: 'h-6 w-6' })
    ),
    React.createElement('h3', { className: 'text-base font-semibold text-rose-300' }, 'Data File Not Found'),
    React.createElement('p', { className: 'mt-2 font-mono text-xs text-rose-400 bg-black/50 px-3 py-1.5 rounded-lg inline-block border border-rose-500/20' }, `data file not found: ${path}`),
    error ? React.createElement('p', { className: 'mt-2 text-xs text-slate-400' }, `Reason: ${error}`) : null,
    React.createElement(
      'div',
      { className: 'mt-4 flex items-center justify-center gap-2 text-xs text-slate-500 font-mono' },
      React.createElement(AlertTriangle, { className: 'h-3.5 w-3.5 text-amber-400 shrink-0' }),
      React.createElement('span', null, 'Rule R7: Missing data is shown as missing. Fabricated values are strictly prohibited.')
    )
  );
}

export function LoadingSpinner({ label = 'Loading verified metrics...' }) {
  return React.createElement(
    'div',
    { className: 'flex flex-col items-center justify-center py-24 space-y-4' },
    React.createElement('div', { className: 'h-8 w-8 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent' }),
    React.createElement('p', { className: 'text-xs font-mono uppercase tracking-widest text-slate-400' }, label)
  );
}
