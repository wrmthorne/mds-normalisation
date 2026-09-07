from __future__ import annotations

import argparse
import asyncio

from experiments.extraction_common import predict
from experiments.variants import REPRESENTATION_VARIANTS

EXP = "representation_variants"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(REPRESENTATION_VARIANTS), required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    asyncio.run(predict(EXP, REPRESENTATION_VARIANTS[args.variant], args.limit))


if __name__ == "__main__":
    main()
