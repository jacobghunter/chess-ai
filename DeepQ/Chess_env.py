import chess
import numpy as np
import random

class ChessEnvironment:
  def __init__(self):
      self.board = chess.Board()
      self.piece_values = {
          'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
          'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0
      }

      # Initialize position matrices
      self.position_matrices = self._initialize_position_matrices()
      self.openings = self._initialize_openings()
      self.move_history = []

  def _initialize_position_matrices(self):
      # Pawn position matrix
      pawn_matrix = np.array([
          [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
          [5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0],
          [1.0,  1.0,  2.0,  3.0,  3.0,  2.0,  1.0,  1.0],
          [0.5,  0.5,  1.0,  2.5,  2.5,  1.0,  0.5,  0.5],
          [0.0,  0.0,  0.0,  2.0,  2.0,  0.0,  0.0,  0.0],
          [0.5, -0.5, -1.0,  0.0,  0.0, -1.0, -0.5,  0.5],
          [0.5,  1.0,  1.0, -2.0, -2.0,  1.0,  1.0,  0.5],
          [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]
      ])

      # Knight position matrix
      knight_matrix = np.array([
          [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0],
          [-4.0, -2.0,  0.0,  0.0,  0.0,  0.0, -2.0, -4.0],
          [-3.0,  0.0,  1.0,  1.5,  1.5,  1.0,  0.0, -3.0],
          [-3.0,  0.5,  1.5,  2.0,  2.0,  1.5,  0.5, -3.0],
          [-3.0,  0.0,  1.5,  2.0,  2.0,  1.5,  0.0, -3.0],
          [-3.0,  0.5,  1.0,  1.5,  1.5,  1.0,  0.5, -3.0],
          [-4.0, -2.0,  0.0,  0.5,  0.5,  0.0, -2.0, -4.0],
          [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0]
      ])

      # Bishop position matrix
      bishop_matrix = np.array([
          [-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0],
          [-1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0],
          [-1.0,  0.0,  0.5,  1.0,  1.0,  0.5,  0.0, -1.0],
          [-1.0,  0.5,  0.5,  1.0,  1.0,  0.5,  0.5, -1.0],
          [-1.0,  0.0,  1.0,  1.0,  1.0,  1.0,  0.0, -1.0],
          [-1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, -1.0],
          [-1.0,  0.5,  0.0,  0.0,  0.0,  0.0,  0.5, -1.0],
          [-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0]
      ])

      # Rook position matrix
      rook_matrix = np.array([
          [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
          [0.5,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.5],
          [-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
          [-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
          [-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
          [-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
          [-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
          [0.0,  0.0,  0.0,  0.5,  0.5,  0.0,  0.0,  0.0]
      ])

      # Queen position matrix
      queen_matrix = np.array([
          [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0],
          [-1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0],
          [-1.0,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0],
          [-0.5,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5],
          [0.0,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5],
          [-1.0,  0.5,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0],
          [-1.0,  0.0,  0.5,  0.0,  0.0,  0.0,  0.0, -1.0],
          [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0]
      ])

      # King middle game position matrix
      king_matrix = np.array([
          [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
          [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
          [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
          [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
          [-2.0, -3.0, -3.0, -4.0, -4.0, -3.0, -3.0, -2.0],
          [-1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0],
          [2.0,  2.0,  0.0,  0.0,  0.0,  0.0,  2.0,  2.0],
          [2.0,  3.0,  1.0,  0.0,  0.0,  1.0,  3.0,  2.0]
      ])

      # King endgame position matrix
      king_endgame_matrix = np.array([
          [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0],
          [-1.0,  0.0,  0.5,  0.0,  0.0,  0.0,  0.0, -1.0],
          [-1.0,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5, -1.0],
          [-0.5,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5],
          [0.0,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5],
          [-1.0,  0.5,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0],
          [-1.0,  0.0,  0.5,  0.0,  0.0,  0.0,  0.0, -1.0],
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
  def _initialize_openings(self):
      return [
          # e4 openings
          ["e2e4", "e7e5"],  # King's Pawn Opening
          ["e2e4", "c7c5"],  # Sicilian Defense
          ["e2e4", "e7e6"],  # French Defense
          ["e2e4", "c7c6"],  # Caro-Kann Defense
          ["e2e4", "d7d6"],  # Pirc Defense

          # d4 openings
          ["d2d4", "d7d5"],  # Queen's Pawn Opening
          ["d2d4", "g8f6"],  # Indian Defense
          ["d2d4", "f7f5"],  # Dutch Defense
          ["d2d4", "e7e6"],  # Queen's Indian Defense

          # Other openings
          ["c2c4"],          # English Opening
          ["g1f3"],          # Réti Opening
          ["b2b3"],          # Larsen's Opening
          ["f2f4"],          # Bird's Opening
      ]

  def get_legal_moves_mask(self):
      """Create a mask of legal moves"""
      mask = np.zeros(4672, dtype=np.float32)
      for move in self.board.legal_moves:
          idx = self._move_to_index(move)
          mask[idx] = 1
      return mask

  def _move_to_index(self, move):
      """Convert a chess move to an index"""
      from_square = move.from_square
      to_square = move.to_square
      # Basic encoding: from_square * 64 + to_square
      return from_square * 64 + to_square

  def _index_to_move(self, index):
      """Convert an index back to a chess move"""
      from_square = index // 64
      to_square = index % 64
      return chess.Move(from_square, to_square)

  def reset(self):
      """Reset the chess board and optionally apply an opening"""
      self.board = chess.Board()
      self.move_history = []

      if random.random() < 0.7:  # 70% chance to use an opening
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
      """Convert current board state to neural network input format"""
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
      """Determine if the current position is an endgame"""
      queens = 0
      minor_pieces = 0

      for square in chess.SQUARES:
          piece = self.board.piece_at(square)
          if piece:
              if piece.piece_type == chess.QUEEN:
                  queens += 1
              elif piece.piece_type in [chess.BISHOP, chess.KNIGHT]:
                  minor_pieces += 1

      return queens == 0 or (queens == 2 and minor_pieces <= 2)

  def _calculate_position_reward(self):
      """Calculate positional reward based on piece positions"""
      reward = 0
      for square in chess.SQUARES:
          piece = self.board.piece_at(square)
          if piece:
              rank, file = divmod(square, 8)
              if not piece.color:  # If black, flip the rank
                  rank = 7 - rank

              piece_type = piece.symbol().upper()
              if piece_type == 'K' and self.is_endgame():
                  matrix = self.position_matrices['K_endgame']
              elif piece_type in self.position_matrices:
                  matrix = self.position_matrices[piece_type]
              else:
                  continue

              value = matrix[rank][file]
              reward += value if piece.color else -value

      return reward

  def _calculate_reward(self):
      """Calculate the total reward for the current position"""
      if self.board.is_checkmate():
          return 100 if not self.board.turn else -100

      if self.board.is_stalemate() or self.board.is_insufficient_material():
          return 0

      # Material reward
      material_reward = sum(self.piece_values.get(piece.symbol(), 0)
                          for square in chess.SQUARES
                          if (piece := self.board.piece_at(square)))

      # Position reward
      position_reward = self._calculate_position_reward()

      # Control of center squares
      center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
      center_control = sum(0.5 if self.board.piece_at(square) and 
                         self.board.piece_at(square).color else 
                         -0.5 if self.board.piece_at(square) else 0
                         for square in center_squares)

      # Mobility reward (number of legal moves)
      mobility = len(list(self.board.legal_moves)) * 0.1

      # Check reward
      check_reward = 0.5 if self.board.is_check() else 0

      # Combine all rewards
      total_reward = (
          material_reward * 1.0 +    # Material value
          position_reward * 0.1 +    # Position value
          center_control * 0.3 +     # Center control
          mobility * 0.1 +           # Mobility
          check_reward              # Check
      )

      return total_reward

  def step(self, action):
      """Execute a move and return the new state, reward, and done flag"""
      legal_moves = list(self.board.legal_moves)
      if not legal_moves:
          return self._get_state(), -100, True

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
      done = self.board.is_game_over()

      # Additional draw conditions
      if len(self.move_history) > 100:  # Prevent very long games
          done = True
          reward = 0

      return new_state, reward, done

  def render(self):
      """Display the current board state"""
      print("\nCurrent board position:")
      print(self.board)
      print(f"\nFEN: {self.board.fen()}")
      print(f"Current evaluation: {self._calculate_reward():.2f}")
