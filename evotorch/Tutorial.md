# Evochess Installation Tutorial
## Requirements
 - Python 3.10.12
 - Nvidia RTX 2080 (Different Drivers Needed For Other Cards)
 - WSL 2.3.26.0
 - GeForce Game Ready Driver 566.14
 - NVIDIA RTX Driver Release 550 R550 U11
 - CUDA Toolkit 12.6 Update 3 For WSL [Install Tutorial](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl)
 - Pip 22.0.2
###
NOTE THAT ANY OTHER INSTALLED LIBRARIES ON PYTHON IS LIKELY TO BREAK THIS, THIS NEEDS A CLEAN PYTHON INSTALATION
## Instalations outside of this Git
- [Muzero-General](https://github.com/werner-duvaud/muzero-general)
- Pip install Requirements.txt
- Pip install muzero_requirements.lock (20+ min)
- [Stockfish Build for Linix](https://stockfishchess.org/download/) for linux
    - Make sure to compile after downloading (20+ min)

## Tutorial
1. Verify all requirements and download all instalations
2. Copy absolute path of stockfish complilation and set ENGINE_PATH in evochess
3. Copy evochess.py into muzerogeneral/games folder
4. cd into muzerogeneral
5. Run python muzero.py
6. Load Results via path
7. If traning note that training takes 48+ hours
