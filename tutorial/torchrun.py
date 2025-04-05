import os

def main():
    # Useful for node communication
    master_address = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    # Useful for training coordination
    rank = os.environ["RANK"]
    local_rank = os.environ["LOCAL_RANK"]
    world_size = os.environ["WORLD_SIZE"]

    # Useful for saving experiment artifacts
    output_path = os.environ["OUTPUT_PATH"]

    print(f"Master address: {master_address}, Master port: {master_port}, Rank: {rank}, Local rank: {local_rank}, World size: {world_size}, Output path: {output_path}")

if __name__ == "__main__":
    main()