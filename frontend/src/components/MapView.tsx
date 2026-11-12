import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { Incident, Responder, Facility } from '../types';

interface MapViewProps {
  incidents?: Incident[];
  responders?: Responder[];
  facilities?: Facility[];
  selectedIncident?: Incident | null;
  routeGeometry?: any;
  center?: [number, number];
  zoom?: number;
  height?: string;
  onSelectIncident?: (incident: Incident) => void;
  onSelectResponder?: (responder: Responder) => void;
}

export const MapView: React.FC<MapViewProps> = ({
  incidents = [],
  responders = [],
  facilities = [],
  selectedIncident = null,
  routeGeometry = null,
  center = [12.9716, 77.5946],
  zoom = 13,
  height = '480px',
  onSelectIncident,
  onSelectResponder,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const layersRef = useRef<{
    incidentsLayer: any;
    respondersLayer: any;
    facilitiesLayer: any;
    routeLayer: any;
  }>({
    incidentsLayer: null,
    respondersLayer: null,
    facilitiesLayer: null,
    routeLayer: null,
  });

  useEffect(() => {
    if (!mapContainerRef.current) return;

    if (!mapInstanceRef.current) {
      const map = L.map(mapContainerRef.current, {
        center,
        zoom,
        zoomControl: true,
      });

      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 19,
      }).addTo(map);

      layersRef.current.incidentsLayer = L.layerGroup().addTo(map);
      layersRef.current.respondersLayer = L.layerGroup().addTo(map);
      layersRef.current.facilitiesLayer = L.layerGroup().addTo(map);
      layersRef.current.routeLayer = L.layerGroup().addTo(map);

      mapInstanceRef.current = map;
    }

    return () => {
      // Map cleanup on unmount
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  // Update center when selectedIncident changes
  useEffect(() => {
    if (mapInstanceRef.current && selectedIncident) {
      mapInstanceRef.current.setView([selectedIncident.latitude, selectedIncident.longitude], 15);
    }
  }, [selectedIncident]);

  // Update Markers
  useEffect(() => {
    const L = (window as any).L;
    if (!L || !mapInstanceRef.current) return;

    const { incidentsLayer, respondersLayer, facilitiesLayer, routeLayer } = layersRef.current;

    // Clear existing
    if (incidentsLayer) incidentsLayer.clearLayers();
    if (respondersLayer) respondersLayer.clearLayers();
    if (facilitiesLayer) facilitiesLayer.clearLayers();
    if (routeLayer) routeLayer.clearLayers();

    // 1. Incidents
    incidents.forEach((inc) => {
      const isCritical = inc.severity === 'CRITICAL';
      const color = isCritical ? '#D95757' : inc.severity === 'HIGH' ? '#F2A23A' : '#E7AE4B';

      const iconHtml = `
        <div style="
          width: 28px;
          height: 28px;
          background: ${color};
          border: 2px solid #FFFFFF;
          border-radius: 50%;
          box-shadow: 0 0 12px ${color};
          display: flex;
          align-items: center;
          justify-content: center;
          color: #090B0A;
          font-weight: 800;
          font-size: 11px;
        ">
          !
        </div>
      `;
      const customIcon = L.divIcon({
        html: iconHtml,
        className: 'incident-marker-icon',
        iconSize: [28, 28],
        iconAnchor: [14, 14],
      });

      const marker = L.marker([inc.latitude, inc.longitude], { icon: customIcon });
      marker.bindPopup(`
        <div style="font-family: inherit; font-size: 12px;">
          <b style="color: ${color}; font-size: 13px;">${inc.incident_id}</b> (${inc.severity})<br/>
          <strong>${inc.label}</strong><br/>
          <span>Status: ${inc.status}</span><br/>
          <span>Camera: ${inc.camera_id}</span>
        </div>
      `);
      if (onSelectIncident) {
        marker.on('click', () => onSelectIncident(inc));
      }
      marker.addTo(incidentsLayer);
    });

    // 2. Responders
    responders.forEach((resp) => {
      const isAvailable = resp.status === 'AVAILABLE';
      const color = isAvailable ? '#55A77B' : '#F2A23A';

      const iconHtml = `
        <div style="
          width: 26px;
          height: 26px;
          background: #111512;
          border: 2px solid ${color};
          border-radius: 6px;
          box-shadow: 0 0 10px ${color};
          display: flex;
          align-items: center;
          justify-content: center;
          color: ${color};
          font-weight: 700;
          font-size: 10px;
        ">
          🚑
        </div>
      `;
      const customIcon = L.divIcon({
        html: iconHtml,
        className: 'responder-marker-icon',
        iconSize: [26, 26],
        iconAnchor: [13, 13],
      });

      const marker = L.marker([resp.latitude, resp.longitude], { icon: customIcon });
      marker.bindPopup(`
        <div style="font-family: inherit; font-size: 12px;">
          <b style="color: ${color}; font-size: 13px;">${resp.name}</b> (${resp.responder_id})<br/>
          <span>Type: ${resp.responder_type}</span><br/>
          <span>Status: <b>${resp.status}</b></span><br/>
          <span>Vehicle: ${resp.vehicle_number || 'N/A'}</span>
        </div>
      `);
      if (onSelectResponder) {
        marker.on('click', () => onSelectResponder(resp));
      }
      marker.addTo(respondersLayer);
    });

    // 3. Hospitals
    facilities.forEach((fac) => {
      const iconHtml = `
        <div style="
          width: 22px;
          height: 22px;
          background: #1D2420;
          border: 1.5px solid #55A77B;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          color: #55A77B;
          font-weight: 800;
          font-size: 11px;
        ">
          H
        </div>
      `;
      const customIcon = L.divIcon({
        html: iconHtml,
        className: 'facility-marker-icon',
        iconSize: [22, 22],
        iconAnchor: [11, 11],
      });

      const marker = L.marker([fac.latitude, fac.longitude], { icon: customIcon });
      const timeInfo = fac.travel_time_minutes ? `<br/><span>Travel Time: <b>${fac.travel_time_minutes} min</b></span>` : '';
      marker.bindPopup(`
        <div style="font-family: inherit; font-size: 12px;">
          <b style="color: #55A77B; font-size: 13px;">${fac.name}</b><br/>
          <span>${fac.address || 'Medical Facility'}</span>${timeInfo}
        </div>
      `);
      marker.addTo(facilitiesLayer);
    });

    // 4. Route
    if (routeGeometry && routeLayer) {
      try {
        const polyline = L.geoJSON(routeGeometry, {
          style: {
            color: '#F2A23A',
            weight: 5,
            opacity: 0.85,
            dashArray: '8, 8',
          },
        });
        polyline.addTo(routeLayer);
      } catch {
        // invalid geometry
      }
    }
  }, [incidents, responders, facilities, routeGeometry]);

  return (
    <div
      ref={mapContainerRef}
      style={{
        width: '100%',
        height,
        borderRadius: '12px',
        border: '1px solid #303832',
        overflow: 'hidden',
        backgroundColor: '#0c0f0d',
      }}
    />
  );
};
