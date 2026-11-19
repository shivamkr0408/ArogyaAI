import React, { createContext, useContext, useEffect, useState } from 'react';

const ThemeContext = createContext<{ dark: boolean; toggle: () => void }>({
  dark: false,
  toggle: () => {},
});

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [dark, setDark] = useState<boolean>(() => {
    const saved = localStorage.getItem('arogya_theme');
    if (saved) return saved === 'dark';
    return window.matchMedia?.('(prefers-color-scheme: dark)')?.matches ?? false;
  });

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
    localStorage.setItem('arogya_theme', dark ? 'dark' : 'light');
  }, [dark]);

  return (
    <ThemeContext.Provider value={{ dark, toggle: () => setDark(d => !d) }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}