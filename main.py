import chess
import chess.svg
import chess.engine
import time
import os
from stockfish import Stockfish


def display():
    board = chess.Board()

    # generates svg image of the board
    boardsvg = chess.svg.board(board, size=600, coordinates=True)

    # in the future we can just have this slotted into some html
    with open('temp.svg', 'w') as outputfile:
        outputfile.write(boardsvg)
    time.sleep(0.1)
    os.startfile('temp.svg')


def stockfish():
    # this library may not be neccesary since chess already has a built in version
    # stockfish1 = Stockfish(
    #     "stockfish\stockfish-windows-x86-64-avx2.exe",
    #     depth=20,  # how many moves deep to look
    #     parameters={
    #         "Contempt": 0,
    #         "Min Split Depth": 0,
    #         # More threads will make the engine stronger, but should be kept at less than the number of logical processors on your computer.
    #         "Threads": 1,
    #         "Ponder": "false",
    #         # Default size is 16 MB. It's recommended that you increase this value, but keep it as some power of 2. E.g., if you're fine using 2 GB of RAM, set Hash to 2048 (11th power of 2).
    #         "Hash": 16,
    #         "MultiPV": 1,
    #         "Skill Level": 20,
    #         "Move Overhead": 10,
    #         "Minimum Thinking Time": 20,
    #         "Slow Mover": 100,
    #         "UCI_Chess960": "false",
    #         "UCI_LimitStrength": "false",
    #         "UCI_Elo": 1350
    #     }
    # )

    engine1 = chess.engine.SimpleEngine.popen_uci(
        "stockfish/stockfish-windows-x86-64-avx2.exe")
    engine2 = chess.engine.SimpleEngine.popen_uci(
        "stockfish/stockfish-windows-x86-64-avx2.exe")

    engine1.configure({
        "Skill Level": 0,  # Skill Level (from 0 to 20)
    })

    engine1.configure({
        "Skill Level": 20,
    })

    board = chess.Board()

    limit = chess.engine.Limit(depth=10)

    print("Starting Position:")
    print(board)

    # Run the game loop where each engine plays one move at a time
    while not board.is_game_over():
        # Stockfish 1 (White) makes a move
        print("Stockfish 1 (White) makes a move:")
        move1 = engine1.play(board, limit).move
        print(f"Move: {move1}")
        board.push(move1)  # Apply move to the board

        # Print the updated board after Stockfish 1's move
        print(board)

        if board.is_game_over():
            break

        # Stockfish 2 (Black) makes a move
        print("Stockfish 2 (Black) makes a move:")
        move2 = engine2.play(board, limit).move
        print(f"Move: {move2}")
        board.push(move2)  # Apply move to the board

        # Print the updated board after Stockfish 2's move
        print(board)

        time.sleep(1)  # Optional: Add a small delay to make it more readable

    print("Game Over")
    print("Result:", board.result())

    # Close the engines after the game ends
    engine1.quit()
    engine2.quit()


if __name__ == "__main__":
    stockfish()
