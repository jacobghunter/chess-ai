import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import numpy as np
import chess
import chess.engine
import matplotlib.pyplot as plt

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


print(f"Using device: {device}")

STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class StockfishEvaluator:
    def __init__(self, path=STOCKFISH_PATH, depth=15, skill_level=20):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        # Attempt to configure stockfish skill level if available
        try:
            self.engine.configure({"Skill Level": skill_level})
        except:
            print("Warning: Could not set Skill Level. Using default.")
        self.depth = depth

    def evaluate(self, board):
        if board.is_game_over():
            if board.is_checkmate():
                return 10000 if board.turn == chess.BLACK else -10000
            else:
                return 0
        result = self.engine.analyse(board, limit=chess.engine.Limit(depth=self.depth))
        score = result['score'].pov(chess.WHITE).score(mate_score=10000)
        return score

    def close(self):
        self.engine.quit()

class ChessNN(nn.Module):
    def __init__(self):
        super(ChessNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(12, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 8 * 8, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 4672)
        )

        self.to(device)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 512 * 8 * 8)
        x = self.fc_layers(x)
        return x

class DeepQLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99995  # Slower decay for more exploration
        self.learning_rate = 5e-5
        self.batch_size = 256  # Increased batch size
        self.model = ChessNN()
        self.target_model = ChessNN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()  # Initialize target model

    def update_target_model(self, tau=0.01):
        # Soft update of target network
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))

    def act(self, state, legal_moves_mask):
        if random.random() < self.epsilon:
            legal_moves = np.where(legal_moves_mask == 1)[0]
            return np.random.choice(legal_moves) if len(legal_moves) > 0 else 0

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy().squeeze()
            q_values[legal_moves_mask == 0] = -np.inf
            return np.argmax(q_values)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(np.array([e.state for e in batch])).to(device)
        actions = torch.LongTensor(np.array([e.action for e in batch])).to(device)
        rewards = torch.FloatTensor(np.array([e.reward for e in batch])).to(device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(device)
        dones = torch.FloatTensor(np.array([e.done for e in batch])).to(device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()

        # Monitor gradient norms
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        # You can log total_norm if needed

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network after each training step
        self.update_target_model()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()

class ChessEnvironment:
    def __init__(self, use_stockfish=False, stockfish_depth=15, stockfish_skill=20):
        self.board = chess.Board()
        self.piece_values = {
            'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
            'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0
        }
        self.position_matrices = self._initialize_position_matrices()
        self.openings = self._initialize_openings()
        self.move_history = []
        self.use_stockfish = use_stockfish
        if self.use_stockfish:
            self.stockfish_evaluator = StockfishEvaluator(depth=stockfish_depth, skill_level=stockfish_skill)

    def _initialize_openings(self):
        return [
            ["e2e4", "e7e5"],
            ["e2e4", "c7c5"],
            ["e2e4", "e7e6"],
            ["e2e4", "c7c6"],
            ["e2e4", "d7d6"],
            ["d2d4", "d7d5"],
            ["d2d4", "g8f6"],
            ["d2d4", "f7f5"],
            ["d2d4", "e7e6"],
            ["c2c4"],
            ["g1f3"],
            ["b2b3"],
            ["f2f4"],
        ]

    def _initialize_position_matrices(self):
        pawn_matrix = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0],
            [0.5, 0.5, 1.0, 2.5, 2.5, 1.0, 0.5, 0.5],
            [0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0],
            [0.5, -0.5, -1.0, 0.0, 0.0, -1.0, -0.5, 0.5],
            [0.5, 1.0, 1.0, -2.0, -2.0, 1.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])

        knight_matrix = np.array([
            [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0],
            [-4.0, -2.0, 0.0, 0.0, 0.0, 0.0, -2.0, -4.0],
            [-3.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -3.0],
            [-3.0, 0.5, 1.5, 2.0, 2.0, 1.5, 0.5, -3.0],
            [-3.0, 0.0, 1.5, 2.0, 2.0, 1.5, 0.0, -3.0],
            [-3.0, 0.5, 1.0, 1.5, 1.5, 1.0, 0.5, -3.0],
            [-4.0, -2.0, 0.0, 0.5, 0.5, 0.0, -2.0, -4.0],
            [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0]
        ])

        bishop_matrix = np.array([
            [-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -1.0],
            [-1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, -1.0],
            [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0],
            [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
            [-1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, -1.0],
            [-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0]
        ])

        rook_matrix = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
            [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
            [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
            [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
            [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
            [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0]
        ])

        queen_matrix = np.array([
            [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0],
            [-0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5],
            [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5],
            [-1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0],
            [-1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -1.0],
            [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0]
        ])

        king_matrix = np.array([
            [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
            [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
            [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
            [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
            [-2.0, -3.0, -3.0, -4.0, -4.0, -3.0, -3.0, -2.0],
            [-1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0],
            [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0],
            [2.0, 3.0, 1.0, 0.0, 0.0, 1.0, 3.0, 2.0]
        ])

        king_endgame_matrix = np.array([
            [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0],
            [-1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -1.0],
            [-1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1.0],
            [-0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5],
            [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5],
            [-1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0],
            [-1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -1.0],
            [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0]
        ])

        return {
            'P': pawn_matrix,
            'N': knight_matrix,
            'B': bishop_matrix,
            'R': rook_matrix,
            'Q': queen_matrix,
            'K': king_matrix,
            'K_endgame': king_endgame_matrix
        }

    def get_legal_moves_mask(self):
        mask = np.zeros(4672, dtype=np.float32)
        for move in self.board.legal_moves:
            idx = self._move_to_index(move)
            mask[idx] = 1
        return mask

    def _move_to_index(self, move):
        from_square = move.from_square
        to_square = move.to_square
        return from_square * 64 + to_square

    def _index_to_move(self, index):
        from_square = index // 64
        to_square = index % 64
        return chess.Move(from_square, to_square)

    def reset(self):
        self.board = chess.Board()
        self.move_history = []
        if random.random() < 0.7:
            opening = random.choice(self.openings)
            for move_uci in opening:
                try:
                    move = chess.Move.from_uci(move_uci)
                    self.board.push(move)
                    self.move_history.append(move)
                except:
                    print(f"Invalid move in opening: {move_uci}")
                    break
        return self._get_state()

    def _get_state(self):
        state = np.zeros((12, 8, 8), dtype=np.float32)
        piece_idx = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                idx = piece_idx[piece.symbol()]
                state[idx][rank][file] = 1
        return state

    def is_endgame(self):
        queens = sum(1 for sq in chess.SQUARES if self.board.piece_at(sq) and self.board.piece_at(sq).piece_type == chess.QUEEN)
        minor_pieces = sum(1 for sq in chess.SQUARES if self.board.piece_at(sq) and self.board.piece_at(sq).piece_type in [chess.BISHOP, chess.KNIGHT])
        return queens == 0 or (queens == 2 and minor_pieces <= 2)

    def _calculate_position_reward(self):
        reward = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                if not piece.color:
                    rank = 7 - rank
                piece_type = piece.symbol().upper()
                if piece_type == 'K' and self.is_endgame():
                    matrix = self.position_matrices['K_endgame']
                else:
                    matrix = self.position_matrices.get(piece_type, None)
                if matrix is None:
                    continue
                value = matrix[rank][file]
                reward += value if piece.color else -value
        return reward

    def _calculate_heuristic_reward(self):
        if self.board.is_checkmate():
            return 100 if not self.board.turn else -100
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0
        material_reward = sum(self.piece_values.get(piece.symbol(), 0)
                              for square in chess.SQUARES
                              if (piece := self.board.piece_at(square)))
        position_reward = self._calculate_position_reward()
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        center_control = sum(0.5 if self.board.piece_at(square) and
                                self.board.piece_at(square).color else
                                -0.5 if self.board.piece_at(square) else 0
                                for square in center_squares)
        mobility = len(list(self.board.legal_moves)) * 0.1
        check_reward = 0.5 if self.board.is_check() else 0
        total_reward = (material_reward * 1.0 +
                        position_reward * 0.1 +
                        center_control * 0.3 +
                        mobility * 0.1 +
                        check_reward)
        return total_reward

    def _calculate_reward(self):
        heuristic = self._calculate_heuristic_reward()
        if self.use_stockfish:
            sf_score = self.stockfish_evaluator.evaluate(self.board)
            sf_norm = np.clip(sf_score / 1000.0, -10, 10) / 10.0
            combined = (0.5 * (heuristic/100.0)) + (0.5 * sf_norm)
            return combined
        else:
            return heuristic / 100.0

    def step(self, action):
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return self._get_state(), -1, True
        try:
            move = self._index_to_move(action)
            if move not in legal_moves:
                move = random.choice(legal_moves)
        except:
            move = random.choice(legal_moves)
        self.board.push(move)
        self.move_history.append(move)
        new_state = self._get_state()
        reward = self._calculate_reward()
        reward = np.clip(reward, -1, 1)
        done = self.board.is_game_over() or len(self.move_history) > 100
        return new_state, reward, done

    def render(self):
        print("\nCurrent board position:")
        print(self.board)
        print(f"\nFEN: {self.board.fen()}")
        print(f"Current evaluation: {self._calculate_reward():.2f}")

    def close(self):
        if self.use_stockfish:
            self.stockfish_evaluator.close()

def board_to_state(board):
    state = np.zeros((12, 8, 8), dtype=np.float32)
    piece_idx = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank, file = divmod(square, 8)
            idx = piece_idx[piece.symbol()]
            state[idx][rank][file] = 1
    return state

def move_to_index(move):
    from_square = move.from_square
    to_square = move.to_square
    return from_square * 64 + to_square

def index_to_move(index):
    from_square = index // 64
    to_square = index % 64
    return chess.Move(from_square, to_square)

def legal_moves_mask_from_board(board):
    mask = np.zeros(4672, dtype=np.float32)
    for move in board.legal_moves:
        idx = move_to_index(move)
        mask[idx] = 1
    return mask

def model_choose_move(agent, state, legal_moves_mask):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = agent.model(state_tensor).cpu().numpy().squeeze()
    q_values[legal_moves_mask == 0] = -np.inf
    return np.argmax(q_values)

def get_suggestions(agent, state, legal_moves_mask, top_n=5):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = agent.model(state_tensor).cpu().numpy().squeeze()
    moves_values = []
    for move_idx in np.where(legal_moves_mask == 1)[0]:
        moves_values.append((move_idx, q_values[move_idx]))
    moves_values.sort(key=lambda x: x[1], reverse=True)
    suggestions = []
    for i, (m_idx, val) in enumerate(moves_values[:top_n]):
        suggestions.append((index_to_move(m_idx).uci(), val))
    return suggestions

def plot_statistics(rewards, lengths, losses):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('rewards.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Number of Moves')
    plt.savefig('episode_lengths.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.savefig('losses.png')
    plt.close()

def train_agent(episodes=10, render_every=100, use_stockfish=False, stockfish_depth=15, stockfish_skill=20, load_existing_weights=False, weights_path='best_model.pth'):
    global total_reward, episode
    if episodes < 100:
        save_every = max(1, episodes // 2)
        render_every = save_every
    else:
        save_every = 100
        render_every = 50

    env = ChessEnvironment(use_stockfish=use_stockfish, stockfish_depth=stockfish_depth, stockfish_skill=stockfish_skill)
    state_size = (12, 8, 8)
    action_size = 4672
    agent = DeepQLearningAgent(state_size, action_size)

    # If requested, try loading existing weights for fine-tuning
    if load_existing_weights and os.path.isfile(weights_path):
        print(f"Loading existing weights from {weights_path} for fine-tuning...")
        checkpoint = torch.load(weights_path, map_location=device)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Weights loaded successfully.")
    else:
        if load_existing_weights:
            print(f"No existing weights found at {weights_path}. Starting from scratch.")

    best_reward = float('-inf')
    episode_rewards = []
    episode_lengths = []
    losses = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        moves = 0
        episode_loss = 0

        while not done and moves < 100:
            if episode % render_every == 0:
                env.render()

            legal_moves_mask = env.get_legal_moves_mask()
            action = agent.act(state, legal_moves_mask)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            loss = agent.replay()
            if loss is not None:
                episode_loss += loss

            state = next_state
            total_reward += reward
            moves += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(moves)
        losses.append(episode_loss / moves if moves > 0 else 0)

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save({
                'episode': episode,
                'model_state_dict': agent.model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'reward': best_reward,
            }, 'best_model.pth')

        if (episode + 1) % save_every == 0:
            torch.save({
                'episode': episode,
                'model_state_dict': agent.model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'reward': total_reward,
            }, f'checkpoint_episode_{episode}.pth')

        print(f"Episode: {episode}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Best Reward: {best_reward:.2f}")
        print(f"Epsilon: {agent.epsilon:.2f}")
        print(f"Moves: {moves}")
        print(f"Average Loss: {losses[-1]:.4f}")
        print("-" * 50)

    torch.save({
        'episode': episode,
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'reward': total_reward,
    }, 'final_model.pth')

    # Plot final statistics
    plot_statistics(episode_rewards, episode_lengths, losses)
    env.close()

def play_with_model(model_path='best_model.pth'):
    state_size = (12, 8, 8)
    action_size = 4672
    agent = DeepQLearningAgent(state_size, action_size)
    checkpoint = torch.load(model_path, map_location=device)
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.model.eval()

    board = chess.Board()
    print("The model (White) will move first. You (Black) will respond.")
    print("Enter your moves in UCI format (e.g., e7e5).")
    print("Press Enter to see suggestions or implement the top suggestion.")
    print("Type 'quit' to exit.\n")

    suggestion_given = False
    suggestions = []

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            print("\nModel's turn (White):")
            state = board_to_state(board)
            legal_moves_mask = legal_moves_mask_from_board(board)
            action = model_choose_move(agent, state, legal_moves_mask)
            model_move = index_to_move(action)
            if model_move not in board.legal_moves:
                model_move = random.choice(list(board.legal_moves))
            print(f"Model plays: {model_move.uci()}")
            board.push(model_move)
            print(board)
        else:
            print("\nYour turn (Black). Enter your move:")
            print(board)
            user_input = input("Your move (uci or press Enter): ").strip()

            if user_input == 'quit':
                break

            if user_input == '':
                if not suggestion_given:
                    state = board_to_state(board)
                    legal_moves_mask = legal_moves_mask_from_board(board)
                    suggestions = get_suggestions(agent, state, legal_moves_mask)
                    print("Model's top suggestions (from White's perspective):")
                    for move_str, val in suggestions:
                        print(f"Move {move_str} with value {val:.4f}")
                    suggestion_given = True
                else:
                    top_suggestion = suggestions[0][0]
                    print(f"Implementing top suggestion: {top_suggestion}")
                    try:
                        user_move = chess.Move.from_uci(top_suggestion)
                        board.push(user_move)
                        suggestion_given = False
                    except Exception as e:
                        print(f"Failed to implement move: {e}")
                continue

            try:
                user_move = chess.Move.from_uci(user_input)
                if user_move not in board.legal_moves:
                    print("Illegal move. Try again.")
                    continue
                board.push(user_move)
                suggestion_given = False
            except:
                print("Invalid move format. Try again.")
                continue

    print("Game over!")
    print(board.result())

def play_with_stockfish(skill_level=20, depth=15):
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    try:
        engine.configure({"Skill Level": skill_level})
    except:
        print("Warning: Could not set Skill Level. Using default.")
    print("Do you want to play as White or Black? (w/b)")
    color = input().strip().lower()
    is_user_white = (color == 'w')

    board = chess.Board()
    print(f"You are {'White' if is_user_white else 'Black'}. Type 'quit' to exit.")

    while not board.is_game_over():
        if (board.turn == chess.WHITE and is_user_white) or (board.turn == chess.BLACK and not is_user_white):
            print("\nYour turn:")
            print(board)
            user_input = input("Your move (uci): ").strip()
            if user_input == 'quit':
                break
            try:
                move = chess.Move.from_uci(user_input)
                if move not in board.legal_moves:
                    print("Illegal move, try again.")
                    continue
                board.push(move)
            except:
                print("Invalid format, try again.")
                continue
        else:
            print("\nStockfish's turn:")
            result = engine.play(board, limit=chess.engine.Limit(depth=depth))
            board.push(result.move)
            print(f"Stockfish plays: {result.move}")
            print(board)

    print("Game over!")
    print(board.result())
    engine.quit()

if __name__ == "__main__":
    while True:
        print("\nChoose an option:")
        print("1. Train a model")
        print("2. Play against a saved model")
        print("3. Exit")
        print("4. Play against Stockfish")

        choice = input("Enter 1, 2, 3, or 4: ").strip()

        if choice == '1':
            episodes = int(input("Number of training episodes: "))
            use_sf = input("Use Stockfish for evaluation? (y/n): ").strip().lower() == 'y'
            sf_skill = 20
            sf_depth = 15
            if use_sf:
                try:
                    sf_skill = int(input("Stockfish skill level (0-20): "))
                except:
                    sf_skill = 20
                try:
                    sf_depth = int(input("Stockfish search depth (e.g., 15): "))
                except:
                    sf_depth = 15

            # Ask if user wants to load existing weights
            load_existing = input("Load existing weights for fine-tuning? (y/n): ").strip().lower() == 'y'
            weights_path = 'best_model.pth'

            train_agent(episodes=episodes, use_stockfish=use_sf, stockfish_depth=sf_depth, stockfish_skill=sf_skill,
                        load_existing_weights=load_existing, weights_path=weights_path)

        elif choice == '2':
            models = [f for f in os.listdir('.') if f.endswith('.pth')]
            if not models:
                print("No model files found.")
            else:
                print("Available model files:")
                for i, m in enumerate(models, start=1):
                    print(f"{i}. {m}")
                while True:
                    model_choice = input("Select a model by number (or type 'cancel' to go back): ").strip()
                    if model_choice == 'cancel':
                        break
                    if model_choice.isdigit() and 1 <= int(model_choice) <= len(models):
                        selected_model = models[int(model_choice)-1]
                        play_with_model(model_path=selected_model)
                        break
                    else:
                        print("Invalid choice. Try again.")

        elif choice == '3':
            print("Exiting.")
            break

        elif choice == '4':
            try:
                sf_skill = int(input("Stockfish skill level (0-20): "))
            except:
                sf_skill = 20
            try:
                sf_depth = int(input("Stockfish search depth (e.g., 15): "))
            except:
                sf_depth = 15
            play_with_stockfish(skill_level=sf_skill, depth=sf_depth)

        else:
            print("Invalid input. Please enter 1, 2, 3, or 4.")