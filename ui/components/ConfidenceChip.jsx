import React from 'react';

const COLORS = {
  low: 'bg-gray-200 text-gray-800',
  medium: 'bg-yellow-100 text-yellow-800',
  high: 'bg-green-100 text-green-800',
};

export default function ConfidenceChip({ tier }) {
  const cls = COLORS[tier] || COLORS.low;
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${cls}`}>{tier}</span>
  );
}

