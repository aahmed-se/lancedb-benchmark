#!/usr/bin/env python

import numpy as np
import time
import lancedb


def generate_random_data_vectors(num_vectors, dimension):
    return [{"vector": list(np.random.random(dimension)), "id": i} for i in range(1, num_vectors + 1)]


def generate_random_vectors(num_vectors, dimension):
    return np.random.random((num_vectors, dimension))


def main():
    # Parameters
    dimension = 128
    num_vectors = int(1000000 / 4)
    num_query_vectors = 20000
    top_k = 10

    db = lancedb.connect("./vectors.db")

    # Generate vectors
    db_vectors = generate_random_data_vectors(num_vectors, dimension)
    query_vectors = generate_random_vectors(num_query_vectors, dimension)

    table = db.create_table("vectors", data=db_vectors)

    print(f"Starting Queries : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    # Perform queries and measure time
    query_times = []
    for i, query_vector in enumerate(query_vectors):
        start_time = time.time()
        table.search(query_vector).limit(top_k)  # Adjust based on actual search method
        end_time = time.time()
        query_duration_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        query_times.append(query_duration_ms)

        # Print time every 5,000 iterations
        if (i + 1) % 5000 == 0:
            print(f"Iteration {i + 1}: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Calculate percentiles
    percentiles = [50, 90, 95, 99]
    percentile_results = {p: np.percentile(query_times, p) for p in percentiles}

    # Print report
    print("Query Time Percentiles (milliseconds):")
    for percentile, value in percentile_results.items():
        print(f"{percentile}th percentile: {value:.3f} ms")

    db.drop_table("vectors")


if __name__ == "__main__":
    main()
