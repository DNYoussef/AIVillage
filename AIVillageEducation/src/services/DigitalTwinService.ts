export interface Interaction {
  transcript: string;
  intent: any;
  timestamp: number;
}

export default class DigitalTwinService {
  private interactions: Interaction[] = [];

  async recordInteraction(interaction: Interaction) {
    this.interactions.push(interaction);
  }

  async getProgress() {
    // Placeholder summary
    return {
      completed: [],
      averageScore: 0,
      totalTime: 0,
      identifiedStrengths: [],
      weakAreas: []
    };
  }
}
