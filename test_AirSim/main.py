# main.py

from utils import setup_logging
from airsim import MultirotorClient
from q_learning import train

def main():
    setup_logging()
    client = MultirotorClient()
    client.confirmConnection()
    print("Connected!")

    drones = ["Drone1"]  # 这里可以根据需要添加更多的无人机
    train(client, drones)

if __name__ == "__main__":
    main()
