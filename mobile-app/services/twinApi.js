const BASE = 'https://api.atlantis.ai';

export default {
  async queryTwin(audioUri) {
    const form = new FormData();
    form.append('file', { uri: audioUri, name: 'rec.wav', type: 'audio/wav' });
    const res = await fetch(`${BASE}/passport/query`, { method: 'POST', body: form });
    return res.json();
  },

  async fetchNextFlashcard() {
    const res = await fetch(`${BASE}/lesson/next`);
    return res.json();
  },

  async gradeAnswer(cardId, audioUri) {
    const form = new FormData();
    form.append('file', { uri: audioUri, name: 'ans.wav', type: 'audio/wav' });
    const res = await fetch(`${BASE}/lesson/${cardId}/grade`, {
      method: 'POST', body: form
    });
    return res.json();
  }
};
