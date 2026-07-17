import { useEffect } from 'react';

type ShortcutMap = {
  [key: string]: (e: KeyboardEvent) => void;
};

export function useKeyboardShortcuts(shortcuts: ShortcutMap) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger if user is typing in an input, textarea, or contenteditable
      const target = e.target as HTMLElement;
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable
      ) {
        return;
      }

      // Handle modifier shortcuts (like Cmd+K) vs single key
      const keyCombo = [
        e.metaKey || e.ctrlKey ? 'cmd+' : '',
        e.shiftKey ? 'shift+' : '',
        e.altKey ? 'alt+' : '',
        e.key.toLowerCase()
      ].join('');

      // Also check exact key (for uppercase single letters like 'U' when Shift is held, though usually better to just check lower)
      const exactKey = e.key;

      if (shortcuts[keyCombo]) {
        e.preventDefault();
        shortcuts[keyCombo](e);
      } else if (shortcuts[exactKey]) {
        e.preventDefault();
        shortcuts[exactKey](e);
      } else if (shortcuts[exactKey.toLowerCase()]) {
        e.preventDefault();
        shortcuts[exactKey.toLowerCase()](e);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts]);
}
