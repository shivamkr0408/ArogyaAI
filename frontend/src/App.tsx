import { ChatProvider } from './hooks/useChatSession';
import { ThemeProvider } from './hooks/useTheme';
import { SpeechProvider } from './hooks/useSpeech';
import ChatWindow from './components/ChatWindow';

export default function App() {
  return (
    <ThemeProvider>
      <SpeechProvider>
        <ChatProvider>
          <ChatWindow />
        </ChatProvider>
      </SpeechProvider>
    </ThemeProvider>
  );
}
