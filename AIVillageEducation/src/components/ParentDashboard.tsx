import React, { Component } from 'react';
import { View, Text } from 'react-native';
import DigitalTwinService from '../services/DigitalTwinService';

interface Report {
  lessonsCompleted: number;
  averageScore: number;
  timeSpent: number;
  strengths: string[];
  areasForImprovement: string[];
  recommendations: string[];
}

export default class ParentDashboard extends Component<{}, { report?: Report }> {
  private digitalTwin: DigitalTwinService;

  constructor(props: {}) {
    super(props);
    this.state = {};
    this.digitalTwin = new DigitalTwinService();
  }

  async componentDidMount() {
    const report = await this.generateReport();
    this.setState({ report });
  }

  async generateReport(): Promise<Report> {
    const progress = await this.digitalTwin.getProgress();
    return {
      lessonsCompleted: progress.completed.length,
      averageScore: progress.averageScore,
      timeSpent: progress.totalTime,
      strengths: progress.identifiedStrengths,
      areasForImprovement: progress.weakAreas,
      recommendations: await this.generateRecommendations(progress)
    };
  }

  async generateRecommendations(progress: any): Promise<string[]> {
    return ['Keep practicing math', 'Review reading lessons'];
  }

  render() {
    const { report } = this.state;
    if (!report) return <Text>Loading...</Text>;
    return (
      <View>
        <Text>Lessons Completed: {report.lessonsCompleted}</Text>
        <Text>Average Score: {report.averageScore}</Text>
        <Text>Total Time: {report.timeSpent}</Text>
        <Text>Strengths: {report.strengths.join(', ')}</Text>
        <Text>Areas: {report.areasForImprovement.join(', ')}</Text>
        <Text>Recommendations: {report.recommendations.join(', ')}</Text>
      </View>
    );
  }
}
