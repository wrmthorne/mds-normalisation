from __future__ import annotations

import argparse
import asyncio

from experiments.extraction_common import predict, predict_mechanical
from experiments.variants import EXTRACTION_VARIANTS

EXP = "extraction_variants"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(EXTRACTION_VARIANTS), required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    variant = EXTRACTION_VARIANTS[args.variant]
    if variant.model is None:
        predict_mechanical(EXP, variant, args.limit)
    else:
        asyncio.run(predict(EXP, variant, args.limit))


if __name__ == "__main__":
    main()
