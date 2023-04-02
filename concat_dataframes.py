#!/usr/bin/env python3

import argparse
import pandas as pd
from typing import List

parser = argparse.ArgumentParser("Concatenate dataframes")
parser.add_argument("--input", "-i", type=str, nargs="+", required=True)
parser.add_argument("--output", "-o", type=str, required=True)


def main(
    inputs: List[str] = tuple(),
    output: str = "",
):
    df = pd.concat([pd.read_csv(f) for f in inputs])
    df.to_csv(output)

    print(f"Concatenated {len(inputs)} dataframes to {output}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        inputs=args.input,
        output=args.output,
    )
