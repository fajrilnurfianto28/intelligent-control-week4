import gym
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent

# Inisialisasi lingkungan dan agen
# render_mode="human" akan menampilkan visualisasi saat pelatihan
env = gym.make("CartPole-v1", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
agent.epsilon = 0.01

# List untuk menyimpan skor dari setiap episode
scores = []
# List untuk menyimpan moving average
moving_avg_scores = []
window_size = 10  # Ukuran jendela untuk moving average

# Loop utama untuk pelatihan agen
# Catatan: Kode ini tidak menguji model, tetapi mengumpulkan skor pelatihan.
for e in range(20):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time in range(500):
        # Env.render() sudah dipanggil di gym.make
        action = agent.act(state)
        # Gunakan format baru untuk gym
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = np.reshape(next_state, [1, state_size])
        total_reward += reward

        if done:
            print(f"Episode {e+1}: Score = {time+1}")
            break

    scores.append(total_reward)
    
    # Hitung moving average
    if len(scores) >= window_size:
        moving_avg = np.mean(scores[-window_size:])
        moving_avg_scores.append(moving_avg)
    else:
        moving_avg_scores.append(np.mean(scores))


env.close()

# --- Bagian untuk menampilkan grafik ---
# Kode ini diletakkan di akhir setelah data 'scores' terisi
plt.plot(moving_avg_scores)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Performance (Moving Average)')
plt.grid(True)
plt.show()