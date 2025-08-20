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
    // Calculate progress summary from recorded interactions
    const completed = this.interactions.map(i => i.intent?.lesson_id).filter(Boolean);
    const uniqueCompleted = [...new Set(completed)];

    const totalTime = this.interactions.reduce((sum, interaction) => {
      return sum + (interaction.intent?.duration || 0);
    }, 0);

    const scores = this.interactions
      .map(i => i.intent?.score)
      .filter(score => typeof score === 'number');

    const averageScore = scores.length > 0
      ? scores.reduce((a, b) => a + b, 0) / scores.length
      : 0;

    return {
      completed: uniqueCompleted,
      averageScore,
      totalTime,
      identifiedStrengths: this.analyzeStrengths(),
      weakAreas: this.analyzeWeaknesses()
    };
  }

  private analyzeStrengths(): string[] {
    // Basic analysis of user strengths based on interactions
    const subjects = this.interactions
      .map(i => i.intent?.subject)
      .filter(Boolean);

    const subjectCounts = subjects.reduce((acc: Record<string, number>, subject) => {
      acc[subject] = (acc[subject] || 0) + 1;
      return acc;
    }, {});

    return Object.entries(subjectCounts)
      .filter(([_, count]) => count >= 3)
      .map(([subject]) => subject);
  }

  private analyzeWeaknesses(): string[] {
    // Basic analysis of areas needing improvement
    const lowScoreSubjects = this.interactions
      .filter(i => i.intent?.score < 0.7)
      .map(i => i.intent?.subject)
      .filter(Boolean);

    return [...new Set(lowScoreSubjects)];
  }
}
