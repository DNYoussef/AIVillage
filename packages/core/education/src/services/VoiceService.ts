import { NativeModules } from 'react-native';

export default class VoiceService {
  private voskModule = NativeModules.VoskVoiceModule;

  async loadModel(name: string) {
    return this.voskModule.loadModel(name);
  }

  async createTTS(config: any) {
    return this.voskModule.createTTS(config);
  }

  startListening(callback: (text: string) => void) {
    this.voskModule.startListening(callback);
  }

  async speak(tts: any, text: string) {
    return this.voskModule.speak(tts, text);
  }
}
