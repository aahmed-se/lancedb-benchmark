#!/usr/bin/env python

import numpy as np
import time
import lancedb
import pyarrow as pa
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_random_data_vectors(num_vectors, dimension, offset=0):
    """
    Generate random data vectors.

    :param num_vectors: Number of vectors to generate.
    :param dimension: Dimension of each vector.
    :param offset: Offset for ID numbering.
    :return: List of dictionaries with vector and id.
    """
    return [{"vector": list(np.random.random(dimension)), "id": i} for i in range(1 + offset, num_vectors + 1 + offset)]


def generate_random_vectors(num_vectors, dimension):
    """
    Generate random vectors using NumPy.

    :param num_vectors: Number of vectors to generate.
    :param dimension: Dimension of each vector.
    :return: NumPy array of random vectors.
    """
    return np.random.random((num_vectors, dimension))


def main():
    # Parameters
    dimension = 1536
    num_vectors = 100000
    num_query_vectors = 20000
    top_k = 10

    db = lancedb.connect("./vectors.db")

    logging.info("Starting Generating Data")
    db_vectors = generate_random_data_vectors(num_vectors, dimension)
    query_vectors = generate_random_vectors(num_query_vectors, dimension)
    logging.info("Finished Generating Data")

    logging.info("Starting to insert data")
    table = db.create_table("vectors", data=db_vectors)
    for i in range(1, 10):
        logging.info(f"Starting to insert data with offset {i}")
        table.add(generate_random_data_vectors(num_vectors, dimension, offset=num_vectors * i))
        logging.info(f"Finished inserting data with offset {i}")

    logging.info("Finished inserting data")

    logging.info("Starting Queries")
    query_times = []
    for i, query_vector in enumerate(query_vectors):
        start_time = time.time()
        table.search(query_vector).limit(top_k)  # Adjust based on actual search method
        end_time = time.time()
        query_duration_ms = (end_time - start_time) * 1000
        query_times.append(query_duration_ms)

        if (i + 1) % 5000 == 0:
            logging.info(f"Iteration {i + 1}")

    percentiles = [50, 90, 95, 99]
    percentile_results = {p: np.percentile(query_times, p) for p in percentiles}

    logging.info("Query Time Percentiles (milliseconds):")
    for percentile, value in percentile_results.items():
        logging.info(f"{percentile}th percentile: {value:.3f} ms")

    db.drop_table("vectors")


if __name__ == "__main__":
    main()
