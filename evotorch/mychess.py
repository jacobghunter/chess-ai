import datetime
import pathlib

import numpy
import torch

import sys

sys.path.append("muzerogeneral/games")
import abstract_game
from abstract_game import AbstractGame

import chess
import chess.engine
import chess.pgn
class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (12, 8, 8)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(4096))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "expert"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 42  # Maximum number of moves if game is not finished before
        self.num_simulations = 200  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        
        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 3  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network
        
        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 100  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.005  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 42  # Number of game moves to keep for every batch element
        self.td_steps = 42  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1
    
def to_actionspace(move: chess.Move):
    return move.to_square+move.from_square*64
    
def from_actionspace(action_space_move: int):
    return chess.Move(action_space_move%64,action_space_move//64)


class Game(AbstractGame):
    def __init__(self, seed=None):
        self.board : chess.Board = chess.Board()
        self.player = 1

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """

        next_move : chess.Move = from_actionspace(action)

        if not self.board.is_legal(next_move):
           next_move = chess.Move(next_move.from_square,next_move.to_square,promotion=chess.QUEEN)
        if not self.board.is_legal(next_move):
            next_move = self.board.generate_legal_moves().__next__()
            print("AUTO MOVE")
        
        print(next_move)
        self.board.push(next_move)

        done = self.board.is_checkmate() or self.board.can_claim_draw() or len(self.legal_actions()) == 0

        reward = 1 if self.board.is_checkmate() else 0

        self.player *= -1

        return self.get_observation(), reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return 0 if self.player == 1 else 1

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        moves = []
        for move in self.board.generate_legal_moves():
            moves.append(to_actionspace(move))
        return moves

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        self.board : chess.Board = chess.Board()
        self.player = 1
        return self.get_observation()
    
    def get_observation(self):
        pawn_white = numpy.array(self.board.pieces(chess.PAWN,chess.WHITE).tolist()).reshape(8,8)
        rook_white = numpy.array(self.board.pieces(chess.ROOK,chess.WHITE).tolist()).reshape(8,8)
        knight_white = numpy.array(self.board.pieces(chess.KNIGHT,chess.WHITE).tolist()).reshape(8,8)
        bisop_white = numpy.array(self.board.pieces(chess.BISHOP,chess.WHITE).tolist()).reshape(8,8)
        queen_white = numpy.array(self.board.pieces(chess.QUEEN,chess.WHITE).tolist()).reshape(8,8)
        king_white = numpy.array(self.board.pieces(chess.KING,chess.WHITE).tolist()).reshape(8,8)

        pawn_black = numpy.array(self.board.pieces(chess.PAWN,chess.BLACK).tolist()).reshape(8,8)
        rook_black = numpy.array(self.board.pieces(chess.ROOK,chess.BLACK).tolist()).reshape(8,8)
        knight_black = numpy.array(self.board.pieces(chess.KNIGHT,chess.BLACK).tolist()).reshape(8,8)
        bisop_black = numpy.array(self.board.pieces(chess.BISHOP,chess.BLACK).tolist()).reshape(8,8)
        queen_black = numpy.array(self.board.pieces(chess.QUEEN,chess.BLACK).tolist()).reshape(8,8)
        king_black = numpy.array(self.board.pieces(chess.KING,chess.BLACK).tolist()).reshape(8,8)
        return numpy.array(
            [pawn_white, 
             rook_white, 
             knight_white, 
             bisop_white,
             queen_white,
             king_white,
             pawn_black, 
             rook_black, 
             knight_black, 
             bisop_black,
             queen_black,
             king_black])

    async def render(self):
        """
        Display the game observation.
        """
        print(chess.svg.board(
            self.board,
            size=350,
        ) )
        await input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        choice = input(f"Enter the standard chess notation moves, here are the legal moves")
        for i in self.board.generate_legal_moves():
            print(i)

        while not self.board.is_legal(chess.Move.from_uci(choice)):
            choice = input("Enter another move : ")
        return to_actionspace(chess.Move.from_uci(choice))

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        result = chess.engine.play(self.board, chess.engine.Limit(
        time=0.0001))
        return to_actionspace(result.move)

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return from_actionspace(action_number).__str__()