import React from 'react';
import ConfidenceChip from './ConfidenceChip';

export default function EvidencePanel({ evidences = [] }) {
  if (!evidences.length) return null;
  return (
    <div className="space-y-1">
      {evidences.map(ev => (
        <div key={ev.id} className="flex items-center space-x-2">
          <ConfidenceChip tier={ev.confidence_tier} />
          <span className="text-sm">{ev.chunks[0].text}</span>
        </div>
      ))}
    </div>
  );
}
