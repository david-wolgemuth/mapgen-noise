#! /usr/bin/env python3

import random
import math
from contextlib import contextmanager
# import named tuple
from collections import namedtuple


Vector2D = namedtuple("Vector2D", ["x", "y"])
Rectangle = namedtuple("Rectangle", ["x0", "y0", "x1", "y1"])


class PerlinNoise2D:
    def __init__(self, seed: str = None):
        self.seed = seed
        self.cache = {}

    def noise(self, x: float, y: float) -> float:
        """
        Returns
        -------
        Value between 0.0 and 1.0
        """
        # determine the vector cell in which the point lies
        left = math.floor(x)
        right = math.floor(x) + 1
        top = math.floor(y)
        bottom = math.floor(y) + 1

        # Distance from point to corners of the vector cell
        tl_dist = Vector2D(x - left, y - top)
        tr_dist = Vector2D(x - right, y - top)
        bl_dist = Vector2D(x - left, y - bottom)
        br_dist = Vector2D(x - right, y - bottom)

        # Get gradients for corners of the vector cell
        tl_grad = self.get_gradient_vector(left, top)
        tr_grad = self.get_gradient_vector(right, top)
        bl_grad = self.get_gradient_vector(left, bottom)
        br_grad = self.get_gradient_vector(right, bottom)

        # Interpolate dot products of gradient vectors and offset vectors
        tl_dot_product = tl_dist.x * tl_grad.x + tl_dist.y * tl_grad.y
        tr_dot_product = tr_dist.x * tr_grad.x + tr_dist.y * tr_grad.y
        bl_dot_product = bl_dist.x * bl_grad.x + bl_dist.y * bl_grad.y
        br_dot_product = br_dist.x * br_grad.x + br_dist.y * br_grad.y

        top_interpolated = self.interpolate(
            tl_dot_product,
            tr_dot_product,
            x - left,
        )
        bottom_interpolated = self.interpolate(
            bl_dot_product,
            br_dot_product,
            x - left,
        )

        # combine top and bottom
        value = self.interpolate(
            top_interpolated,
            bottom_interpolated,
            y - top,
        )

        value = value * 0.5 + 0.5  # scale to 0.0 - 1.0
        if value < 0 or value > 1:
            __import__("ipdb").set_trace()  # FIXME
        return value

    def get_gradient_vector(self, x: int, y: int) -> Vector2D:
        if vector := self.cache.get(f"{x},{y}"):
            return vector

        with self.random_seed_state(f"{x},{y}"):
            self.cache[f"{x},{y}"] = Vector2D(
                x=random.uniform(-1, 1),
                y=random.uniform(-1, 1),
            )
            return self.cache[f"{x},{y}"]

    def interpolate(self, a, b, weight):
        """
        https://en.wikipedia.org/wiki/Linear_interpolation
        https://en.wikipedia.org/wiki/Smoothstep
        """
        if weight < 0:
            # (clamp)
            return a
        if weight > 1:
            # (clamp)
            return b
        weight = self.smooth_step_5o(weight)
        # return (1 - weight) * a + weight * b
        return (b - a) * weight + a

    def smooth_step_5o(self, x):
        """
        fifth-order "smootherstep" function
        https://en.wikipedia.org/wiki/Smoothstep
        """
        return (6 * math.pow(x, 5)) - (15 * math.pow(x, 4)) + (10 * math.pow(x, 3))

    @contextmanager
    def random_seed_state(self, value: str):
        if self.seed:
            random_state = random.getstate()
            random.seed(f"{args.seed}{value}")

        yield

        if self.seed:
            random.setstate(random_state)


def generate_random_matrix(
    rows: int,
    columns: int,
    frequency: float = 1.0,
    offset: float = 0.0,
    octaves: int = 1,
    multiplier: float = 1.0,
    seed: int = None,
    exponent: float = 1.0,
) -> list[list[float]]:
    """
    Generate a matrix of noise values.
    """
    noise_matrix = []
    noise = PerlinNoise2D(seed=seed)
    for y in range(rows):
        row = []
        for x in range(columns):

            xp = x / columns
            yp = y / rows

            value = 0
            freq = frequency
            amp = 1.0

            amplitudes = []
            # https://www.redblobgames.com/maps/terrain-from-noise/#octaves
            for _ in range(octaves):
                value += noise.noise(
                    xp * freq,
                    yp * freq,
                ) * amp

                amplitudes.append(amp)
                amp /= 2
                freq *= 2

            value = value / sum(amplitudes)
            value = math.pow(value * multiplier, exponent)

            value += offset
            if value < 0:  # clamp
                value = 0
            elif value > 1:
                value = 0.999999

            row.append(value)
        noise_matrix.append(row)
    return noise_matrix


def visualize_matrix(matrix):
    RAINBOW = ["🟥", "🟧", "🟨", "🟩", "🟦", "🟪"]
    for row in matrix:
        for value in row:
            if value < 0:
                raise ValueError(f"Value {value} is less than 0")
            color_index = math.floor(value * len(RAINBOW))
            try:
                print(RAINBOW[color_index], end="")
            except IndexError as e:
                print(value)
                print(color_index)
                raise e
        print()


def csv_matrix(matrix):
    for row in matrix:
        print(",".join([f"{value:.2f}" for value in row]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="""Generate Random 2D Matrix

        Known good values:
        ./main.py --columns=160 --rows=80 --frequency=2 --octaves=3 --exponent=1.8 --offset=-1.2 --multiplier=2.5 --seed="hello-world"
        """
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
        "--seed",
        type=str,
        default=None,
        help="Random seed"
    )

    parser.add_argument(
        "--frequency",
        type=float,
        default=1.0,
        help="Zoom factor / number of gradients"
    )

    parser.add_argument(
        "--octaves",
        type=int,
        default=1,
        help="Number of octaves"
    )

    parser.add_argument(
        "--exponent",
        type=float,
        default=1.0,
        help="Exponent, best between 0.0 and 2.0"
    )

    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Offset, best between -1.0 and 1.0"
    )

    parser.add_argument(
        "--multiplier",
        type=float,
        default=1.0,
        help="Multiplier, best between 0.0 and 2.0"
    )

    parser.add_argument(
        "-f", "--format",
        type=str,
        default="visual",
        choices=["csv", "json", "visual"],
        help="Output format"
    )

    args = parser.parse_args()

    matrix = generate_random_matrix(
        rows=args.rows,
        columns=args.columns,
        frequency=args.frequency,
        seed=args.seed,
        octaves=args.octaves,
        multiplier=args.multiplier,
        offset=args.offset,
        exponent=args.exponent,
    )

    if args.format == "visual":
        visualize_matrix(matrix)
    elif args.format == "csv":
        csv_matrix(matrix)
    else:
        raise NotImplementedError(f"Format {args.format} not implemented")
