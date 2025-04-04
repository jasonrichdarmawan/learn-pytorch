import os

def main():
    # Useful for node communication
    master_address = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    # Useful for training coordination
    rank = os.environ["RANK"]

    # Useful for saving experiment artefacts
    output_path = os.environ["OUTPUT_PATH"]

    print(f"Master address: {master_address}, Master port: {master_port}, Rank: {rank}, Output path: {output_path}")

if __name__ == "__main__":
    main()