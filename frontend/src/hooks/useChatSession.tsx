import React, { createContext, useContext, useState, useCallback, useRef, useEffect, ReactNode } from 'react';
import { api, ChatResponse, ChatHistoryResponse } from '../api/client';
import { useHistory, ConsultationRecord } from './useHistory';
import { useProfile } from './useProfile';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  metadata_?: unknown;
}

interface ChatContextType {
  sessionId: string | null;
  messages: Message[];
  state: string;
  loading: boolean;
  send: (message: string) => Promise<void>;
  startNew: () => Promise<void>;
  /** Past consultations kept on this device, most recent first. */
  history: ConsultationRecord[];
  /** Reopen a past consultation by id; the transcript is re-fetched. */
  openSession: (id: string) => Promise<void>;
  removeFromHistory: (id: string) => void;
  clearHistory: () => void;
}

const ChatContext = createContext<ChatContextType | null>(null);

export function useChat() {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error('useChat must be used within ChatProvider');
  return ctx;
}

function getSessionId(): string | null {
  return localStorage.getItem('arogya_session_id');
}

function setSessionId(id: string) {
  localStorage.setItem('arogya_session_id', id);
}

export function ChatProvider({ children }: { children: ReactNode }) {
  const [sessionId, setSession] = useState<string | null>(getSessionId);
  const [messages, setMessages] = useState<Message[]>([]);
  const [state, setState] = useState<string>('greeting');
  const [loading, setLoading] = useState(false);
  const initialized = useRef(false);
  const { forRequest: profileForRequest } = useProfile();
  const {
    records,
    recordTurn,
    remove: removeFromHistory,
    clear: clearHistory,
  } = useHistory();

  const initSession = useCallback(async () => {
    let sid = getSessionId();
    if (sid) {
      try {
        const history: ChatHistoryResponse = await api.getHistory(sid);
        setSession(sid);
        setMessages(history.messages.map(m => ({ role: m.role as 'user' | 'assistant', content: m.content, metadata_: m.metadata_ })));
        setState(history.state);
        return;
      } catch {
        localStorage.removeItem('arogya_session_id');
      }
    }
    const { session_id } = await api.createSession();
    setSessionId(session_id);
    setSession(session_id);
    setState('greeting');
    setMessages([]);
  }, []);

  useEffect(() => {
    if (!initialized.current) {
      initialized.current = true;
      initSession();
    }
  }, [initSession]);

  const send = useCallback(async (message: string) => {
    if (!sessionId || loading) return;
    setLoading(true);
    // The first thing the user says becomes the title of the consultation
    // in the history list, so capture it before the reply comes back.
    const isFirst = !messages.some(m => m.role === 'user');
    setMessages(prev => [...prev, { role: 'user', content: message }]);
    try {
      const res: ChatResponse = await api.sendChat(sessionId, message, profileForRequest());
      setMessages(prev => [...prev, { role: 'assistant', content: res.reply_text, metadata_: res }]);
      setState(res.state);
      recordTurn(sessionId, res, isFirst ? message : undefined);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, something went wrong. Please try again.' }]);
    } finally {
      setLoading(false);
    }
  }, [sessionId, loading, messages, recordTurn, profileForRequest]);

  const startNew = useCallback(async () => {
    const { session_id } = await api.createSession();
    setSessionId(session_id);
    setSession(session_id);
    setMessages([]);
    setState('greeting');
  }, []);

  const openSession = useCallback(async (id: string) => {
    setLoading(true);
    try {
      const history: ChatHistoryResponse = await api.getHistory(id);
      setSessionId(id);
      setSession(id);
      setMessages(history.messages.map(m => ({
        role: m.role as 'user' | 'assistant',
        content: m.content,
        metadata_: m.metadata_,
      })));
      setState(history.state);
    } catch {
      // The backend no longer has it (database reset, say). Drop the dead
      // entry rather than leaving a row that cannot be opened.
      removeFromHistory(id);
    } finally {
      setLoading(false);
    }
  }, [removeFromHistory]);

  return (
    <ChatContext.Provider value={{
      sessionId, messages, state, loading, send, startNew,
      history: records, openSession, removeFromHistory, clearHistory,
    }}>
      {children}
    </ChatContext.Provider>
  );
}
