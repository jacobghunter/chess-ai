import matplotlib.pyplot as plt

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