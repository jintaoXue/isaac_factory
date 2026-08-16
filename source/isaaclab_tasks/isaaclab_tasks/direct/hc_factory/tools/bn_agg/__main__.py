"""Stage-C aggregation: window features, labels, events, job KPI.

From ``tools/``::

    python -m bn_agg --run_dir ../output/bottleneck_dataset/<run_id> \\
        --window_sizes 30,60 --horizon 180

While Isaac is still collecting (another tmux)::

    python -m bn_agg --run_dir ../output/bottleneck_dataset/<run_id> --follow
"""

from .pipeline import main

if __name__ == "__main__":
    main()
