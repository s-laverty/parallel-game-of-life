# Parallel Conway's Game of Life

This project implements Conway's game of life as a scalable parallel task.

To compile, use `build.sh`. This will run our Makefile to produce two builds: `simulation-debug`, a debug build, and `simulation`, an optimized build.

## Usage

This program should be run through MPI. Here is a brief overview of our CLI:
```
Usage: ./simulation [-l checkpoint] [-o checkpoint] [-i hardcode_initializer] [-s grid_size] [-w] strategy num_steps

Required args:
strategy                one of "striped", "brick", or "pipeline"
num_steps               nonnegative integer, number of simulation steps to run

Optional args:
-l checkpoint           filename of a checkpoint to initialize the simulation. Mutually exclusive with the -i option
-o checkpoint           filename specifying where to save the final state of the simulation.
-i hardcode_initializer one of "acorn", "beacon", "beehive", "glider", or "traffic_light".
-s grid_size            one of "s", "m", "l", "xl".
-w                      if provided, use weak scaling (grid size is scaled with number of ranks).
```

The bash script `batch.sh` will automatically batch a comprehensive suite of scaling performance tests by calling `job.sh`, where examples using this CLI can be found.
