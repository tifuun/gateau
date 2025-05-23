# High prioriy

# Mid priority

# Low priority
make GalSpec work -> maybe think abput how sources are implemented.

# Louis filters
## Discuss more in-depth with Louis and Akira maybe (if Louis interested)
- Matrix for filter responses: n x m (m: psd for input frequencies, n: output power for filter frequencies.)
    - Output of main simulation should be psd! -> R not required for main simulation.
- About 4000 freqs between 220 - 440 GHz.
- Consider moving instrument efficiency into the response matrix.
- Parallelise matrix multiplication (multithread, can also do GPU (Nikita?)).
- Think about how to calculate sensitivity. 
    - Maybe look again at sensitivity excercise, but replace the step where we multiply with f / R by the matrix multiplication.
    - How about: calculate P(nu) for each filter. This is the actual information we obtain from an observation. Do for each chop path.
    - Each chop gives P0(nu) (ON) and P1(nu) (OFF).
    - Convert each P0 and P1 to N0 and N1: N = P / (h x nu) x t_obs
    - Definately use the filtered power for GR noise.
- Implement Cramer-Rao bound for calculating variance of filtered signals.
    - Think about what this bound means. Can reconstruct continuous signal with certain variance as function of frequency, from binned and filtered signal?
    - Certainly, if some big matrix multiplication, has to be done parallel as well then.

- Of course, keep option with R and option -> requires less knowledge of actual filter shape.
- Implies that this type of simulation can also be used to test noise removal procedures and filterbank design
    - Can publish the new code separately then.

