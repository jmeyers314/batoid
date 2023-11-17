Changes from v0.4 to v0.5
=========================


API Changes
-----------


New Features
------------
- Default to 1 thread, unless OMP_NUM_THREADS is set.
- Add active optics telescope configurations to parallel_trace_timing.py
- Add ghosts notebook
- Add Rubin v3.12 optics description
- Add Optic.withInsertedOptic and Optic.withRemovedOptic
- Add Optic.R_outer and R_inner properties


Performance Improvements
------------------------
- Add global variable _batoid_niter to control number of iterations
  when solving for intersection.


Bug Fixes
---------
- Allow random seeds in applicable RayVector contructors.
- Trap divide-by-zero error in field_to_dircos
- Clean up CompoundOptic cached properties when inserting/removing items
