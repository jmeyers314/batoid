import os

# Respect OMP_NUM_THREADS, but default to 1 if not present.
_batoid_max_threads = int(os.environ.get('OMP_NUM_THREADS', 1))

# Number of iterations to use in root finding.
# 3 was found to be sufficient for unit tests
# 5 is therefore fairly conservative.
_batoid_niter = 5
