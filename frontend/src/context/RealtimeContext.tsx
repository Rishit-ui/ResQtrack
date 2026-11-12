import React, { createContext, useContext, useEffect, useState, useRef } from 'react';
import { useAuth } from './AuthContext';

interface RealtimeContextType {
  status: 'connected' | 'connecting' | 'disconnected';
  lastEvent: any;
  events: any[];
  followIncident: (id: string) => void;
  unfollowIncident: (id: string) => void;
  playAlertTone: () => void;
}

const RealtimeContext = createContext<RealtimeContextType | undefined>(undefined);

export const RealtimeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, token } = useAuth();
  const [status, setStatus] = useState<'connected' | 'connecting' | 'disconnected'>('disconnected');
  const [lastEvent, setLastEvent] = useState<any>(null);
  const [events, setEvents] = useState<any[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  const playAlertTone = () => {
    try {
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(880, audioCtx.currentTime); // A5
      osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.3);
      gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.35);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.35);
    } catch {
      // AudioContext not permitted without user gesture or unsupported
    }
  };

  useEffect(() => {
    let unmounted = false;

    function connectWs() {
      if (unmounted) return;
      setStatus('connecting');

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host;
      const roleParam = user?.role === 'ADMIN' ? 'control' : user?.role === 'RESPONDER' ? 'responder' : 'citizen';
      const responderIdParam = user?.responder_id || '';
      const tokenParam = token || '';

      const url = `${protocol}//${host}/ws?role=${roleParam}&responder_id=${responderIdParam}&token=${encodeURIComponent(tokenParam)}`;

      try {
        const socket = new WebSocket(url);
        wsRef.current = socket;

        socket.onopen = () => {
          if (!unmounted) setStatus('connected');
        };

        socket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            setLastEvent(data);
            setEvents((prev) => [data, ...prev.slice(0, 49)]);

            // If an incident is created or assigned, trigger tone
            if (data.type === 'incident.alert' || data.type === 'incident.created' || data.type === 'report.created') {
              playAlertTone();
            }
          } catch {
            // non-json message
          }
        };

        socket.onclose = () => {
          if (!unmounted) {
            setStatus('disconnected');
            // Try reconnecting in 3 seconds
            reconnectTimeoutRef.current = window.setTimeout(connectWs, 3000);
          }
        };

        socket.onerror = () => {
          socket.close();
        };
      } catch {
        if (!unmounted) {
          setStatus('disconnected');
          reconnectTimeoutRef.current = window.setTimeout(connectWs, 4000);
        }
      }
    }

    connectWs();

    return () => {
      unmounted = true;
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [user?.user_id, token]);

  const followIncident = (incidentId: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'follow', incident_id: incidentId }));
    }
  };

  const unfollowIncident = (incidentId: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'unfollow', incident_id: incidentId }));
    }
  };

  return (
    <RealtimeContext.Provider
      value={{
        status,
        lastEvent,
        events,
        followIncident,
        unfollowIncident,
        playAlertTone,
      }}
    >
      {children}
    </RealtimeContext.Provider>
  );
};

export const useRealtime = () => {
  const ctx = useContext(RealtimeContext);
  if (!ctx) throw new Error('useRealtime must be used within RealtimeProvider');
  return ctx;
};
