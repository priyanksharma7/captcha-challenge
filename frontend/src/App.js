import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './App.css';

const BASE_URL = 'https://captcha-solver-kbqd.onrender.com';
const NUM_IMAGES = 8;

function formatTime(ms) {
  const totalSec = Math.floor(ms / 1000);
  const minutes = String(Math.floor(totalSec / 60)).padStart(2, '0');
  const seconds = String(totalSec % 60).padStart(2, '0');
  return `${minutes}:${seconds}`;
}

function App() {
  const [username, setUsername] = useState('');
  const [started, setStarted] = useState(false);
  const [finished, setFinished] = useState(false);
  const [cleanImages, setCleanImages] = useState([]);
  const [noisyImages, setNoisyImages] = useState([]);
  const [truths, setTruths] = useState([]);
  const [userInputs, setUserInputs] = useState([]);
  const [aiAnswers, setAiAnswers] = useState([]);
  const [userStartTime, setUserStartTime] = useState(null);
  const [userElapsed, setUserElapsed] = useState(0);
  const [aiTime, setAiTime] = useState(0);
  const [userScore, setUserScore] = useState(0);
  const [aiScore, setAiScore] = useState(0);
  const [leaderboard, setLeaderboard] = useState([]);
  const [showResults, setShowResults] = useState(false);
  const [showInstructions, setShowInstructions] = useState(true);

  useEffect(() => {
    fetchLeaderboard();
    preloadCaptchas();
  }, []);

  useEffect(() => {
    let interval;
    if (started && !finished && userStartTime) {
      interval = setInterval(() => {
        setUserElapsed(Date.now() - userStartTime);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [started, finished, userStartTime]);

  const preloadCaptchas = async () => {
    const res = await fetch(`${BASE_URL}/generate?num_images=${NUM_IMAGES}`);
    const data = await res.json();
    if (!data || !Array.isArray(data.noisy_images) || !Array.isArray(data.clean_images) || !Array.isArray(data.truths)) {
      alert("Error: Couldn't load CAPTCHAs. Please try again.");
      return;
    }
    setNoisyImages(data.noisy_images);
    setCleanImages(data.clean_images);
    setTruths(data.truths);
    setUserInputs(Array(NUM_IMAGES).fill(''));
    setAiAnswers(Array(NUM_IMAGES).fill('Thinking...'));
  };

  const fetchLeaderboard = async () => {
    try {
      const res = await fetch(`${BASE_URL}/leaderboard`);
      const data = await res.json();
      setLeaderboard(data);
    } catch (err) {
      console.error('Leaderboard fetch failed', err);
    }
  };

  const handleStart = () => {
    if (!username.trim()) return alert('Please enter your name');
    setStarted(true);
  };

  const dismissInstructions = async () => {
    setShowInstructions(false);
    setUserStartTime(Date.now());
    const aiStart = Date.now();
    const promises = cleanImages.map((img, idx) => {
      return fetch(`${BASE_URL}/solve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: img })
      })
        .then(res => res.json())
        .then(data => {
          setAiAnswers(prev => {
            const updated = [...prev];
            updated[idx] = data.answer;
            return updated;
          });
        })
        .catch(() => {
          setAiAnswers(prev => {
            const updated = [...prev];
            updated[idx] = 'Error';
            return updated;
          });
        });
    });
    Promise.all(promises).then(() => setAiTime(Date.now() - aiStart));
  };

  const handleInputChange = (idx, value) => {
    const updated = [...userInputs];
    updated[idx] = value;
    setUserInputs(updated);
  };

  const handleSubmit = async () => {
    const elapsed = Date.now() - userStartTime;
    setUserElapsed(elapsed);
    setFinished(true);
    setShowResults(true);

    let userCorrect = 0;
    let aiCorrect = 0;
    for (let i = 0; i < NUM_IMAGES; i++) {
      const truth = truths[i];
      if (userInputs[i] === truth) userCorrect++;
      if (aiAnswers[i] === truth) aiCorrect++;
    }
    setUserScore(userCorrect);
    setAiScore(aiCorrect);

    await fetch(`${BASE_URL}/store`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username,
        userscore: userCorrect,
        usertime: elapsed / 1000,
        ai_score: aiCorrect,
        ai_time: aiTime / 1000
      })
    });
    fetchLeaderboard();
  };

  const handlePlayAgain = () => {
    setStarted(false);
    setFinished(false);
    setShowResults(false);
    setUserElapsed(0);
    setShowInstructions(true);
    preloadCaptchas();
  };

  const determineWinner = () => {
    if (userScore > aiScore) return '🎉 You win!';
    if (aiScore > userScore) return '🤖 AI wins!';
    return userElapsed < aiTime ? '🎉 You win (faster)!' : '🤖 AI wins (faster)!';
  };

  return (
    <div className="app">
      <AnimatePresence>
        {!started && !finished && (
          <motion.div className="intro" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <h1>CAPTCHA Challenge</h1>
            <p>Can you beat an AI model in solving CAPTCHAs?🧠</p>
            <div className="input-start-group">
              <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} placeholder="Enter your name" className="input" />
              <button onClick={handleStart} disabled={!username} className="button sleek">Start the Challenge</button>
            </div>
            <h2>🏆 Leaderboard</h2>
            <div className="leaderboard" style={{ maxWidth: '500px', margin: '0 auto' }}>
              <table>
                <thead><tr><th>Name</th><th>Score</th><th>Time</th></tr></thead>
                <tbody>{leaderboard.map((entry, idx) => (<tr key={idx}><td>{entry.username}</td><td>{entry.userscore}</td><td>{formatTime(entry.usertime * 1000)}</td></tr>))}</tbody>
              </table>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {started && (
        <div className="challenge">
          <div className="challenge-header">
            <h2>Challenge in Progress</h2>
            <div className="timer">⏱️ {formatTime(userElapsed)}</div>
          </div>

          <AnimatePresence>
            {showInstructions && (
              <motion.div className="backdrop" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <motion.div className="popup-card" initial={{ scale: 0.9 }} animate={{ scale: 1 }} transition={{ type: 'spring' }}>
                  <h2>🎯 Instructions</h2>
                  <p>This page will randomly generate 8 CAPTCHAs. Type in your answers as quickly and accurately as possible. You're racing against an AI - whoever gets more correct answers wins. In case of a tie, the faster one takes the crown 👑<br />
                  <i>Pro tip: Use the Tab key to quickly navigate to the next textbox.</i>
                </p>
                  <button className="button green sleek" onClick={dismissInstructions}>Let's go!</button>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>

          <div className="captcha-grid" style={{ gridTemplateRows: `repeat(${Math.ceil(NUM_IMAGES / 4)}, 1fr)` }}>
            {noisyImages.map((img, idx) => (
              <motion.div key={idx} className="captcha-card" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <img src={img} alt={`captcha-${idx + 1}`} />
                <input type="text" value={userInputs[idx]} onChange={(e) => handleInputChange(idx, e.target.value)} placeholder="Your answer" />
                <p className="ai-text">AI: {aiAnswers[idx]}</p>
              </motion.div>
            ))}
          </div>

          {!finished && <div className="center"><button onClick={handleSubmit} className="button green sleek">Submit Answers</button></div>}
        </div>
      )}

      <AnimatePresence>
        {finished && showResults && (
          <motion.div className="backdrop" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <motion.div className="popup-card" initial={{ scale: 0.9 }} animate={{ scale: 1 }} transition={{ type: 'spring' }}>
              <h2>🏁 Final Results</h2>
              <div className="results-card">
                <div style={{ display: 'flex', justifyContent: 'space-around', fontSize: '1.2rem' }}>
                  <div><strong>You</strong><br/>Score: {userScore}<br/>Time: {formatTime(userElapsed)}</div>
                  <div><strong>AI</strong><br/>Score: {aiScore}<br/>Time: {formatTime(aiTime)}</div>
                </div>
              </div>
              <h3 className="winner">{determineWinner()}</h3>
              <button className="button yellow sleek" onClick={handlePlayAgain}>Play Again</button>
          <p style={{ marginTop: '1rem', fontSize: '0.9rem' }}>
              <a href="https://github.com/priyanksharma7/captcha-challenge" target="_blank" rel="noopener noreferrer" style={{ color: '#60a5fa', textDecoration: 'underline' }}>Source Code🧾</a>
            </p>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;