#! /usr/bin/env python3

import random
import math
import json
import csv


def generate_random_matrix(rows, columns):
    matrix = []
    for _ in range(rows):
        row = []
        for _ in range(columns):
            row.append(random.random())
        matrix.append(row)

    return matrix


def visualize_matrix(matrix):
    RAINBOW = ["🟥", "🟧", "🟨", "🟩", "🟦", "🟪"]
    for row in matrix:
        for value in row:
            color_index = math.floor(value * len(RAINBOW))
            print(RAINBOW[color_index], end="")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Random 2D Matrix"
    )

    parser.add_argument(
        "-r", "--rows",
        type=int,
        default=64,
        help="Number of rows in the matrix"
    )

    parser.add_argument(
        "-c", "--columns",
        type=int,
        default=64,
        help="Number of columns in the matrix"
    )

    parser.add_argument(
        "--octaves",
        type=float,
        default=1.0,
        help="Number of octaves"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )

    parser.add_argument(
        "-f", "--format",
        type=str,
        default="visual",
        choices=["csv", "json", "visual"],
        help="Output format"
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    matrix = generate_random_matrix(args.rows, args.columns)

    if args.format == "visual":
        visualize_matrix(matrix)
    else:
        raise NotImplementedError(f"Format {args.format} not implemented")

