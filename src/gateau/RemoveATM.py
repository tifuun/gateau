import numpy as np
import math
import matplotlib.pyplot as pt
import copy

def avgDirectSubtract_spectrum(output):
    """!
    Calculate a spectrum by averaging over the time-domain signal.
    
    Atmosphere removal is done by direct subtraction, using the ON-OFF or ABBA scheme.
    Both schemes will work using this single method.

    @param output Output object obtained from a simulation.

    @returns spectrum Averaged and direct-subtracted spectrum.
    """
   
    # Determine total length of each timestream to not go out of index
    n_samps = output.get("signal").shape[0]

    n_eff = 0

    # We define two counters: 
    #   - pos_old
    #   - pos_now
    # The first keeps track of the chop position of the previous timestamp.
    # The last of the current timestamp. 
    # This is done to determine in which local average the spectrum goes.
    pos_old = 0
    pos_now = 0

    # Container for storing the local average.
    # Local, as in, for storing the average in a single chop position.
    loc_avg = np.zeros(output.get("signal").shape[1])
    
    # Two containers, each for the stored average in a chopping path during an A or B.
    # When pos_now changes w.r.t. pos_old, the local average container is averaged over the total timepoints.
    # The result is either stored in ON_container (0 and 2) or OFF_container (1 and -1).
    ON_container = np.zeros(output.get("signal").shape[1])
    OFF_container = np.zeros(output.get("signal").shape[1])

    # Container for storing the difference between two subsequent ON-OFF averages.
    # This is added to when a full ON-OFF cycle has been performed.
    diff_container = np.zeros(output.get("signal").shape[1])

    # Total average over observation
    tot_avg = np.zeros(output.get("signal").shape[1])

    # Counter for how many points to average over when averaging inside a single chop position.
    n_in_chop = 0

    # Two counters for how many ON and OFF positions have been recorded within a nod position.
    # If n_ON > n_OFF (n_OFF > n_ON) at the end of a nod position, the final ON (OFF) local average will be discarded.
    n_ON = 0
    n_OFF = 0

    # Trackers of old nod position and current one.
    # This is updated and checked by a lambda whenever the pos_now != pos_old
    check_nod = lambda x : "A" if ((x == 0) or (x == 1)) else "B"

    nod_old = "A"
    nod_now = "A"

    # Total number of chopping nods, as a check
    n_nod_tot_ch = 0

    for i in range(n_samps):
        # Start by updating current position
        pos_now = output.get("flag")[i]
       
        nod_now = check_nod(pos_now)
        
        # Just keep storing in loc_avg
        if pos_now == pos_old:
            loc_avg += output.get("signal")[i,:]
            
            n_in_chop += 1

            n_eff += 1

        # Encountered new chop position.
        else:
            
            # Calculate average over loc_avg in previous chop and store in appropriate container.
            # Also update approriate counter
            if ((pos_old == 0) or (pos_old == 2)):
                ON_container += loc_avg / n_in_chop
                n_ON += 1
            
            elif ((pos_old == 1) or (pos_old == -1)): 
                OFF_container += loc_avg / n_in_chop
                n_OFF += 1
            

            # Check if, by updating the numbers, a pair of ON-OFF averages has been obtained
            if n_ON == n_OFF:
                diff_container += ON_container - OFF_container
                ON_container.fill(0)
                OFF_container.fill(0)
                n_nod_tot_ch += 1

            # Check if also a new nod position has been entered by the new chop
            if nod_old != nod_now:
                ON_container.fill(0)
                OFF_container.fill(0)
                n_ON = 0
                n_OFF = 0
                

            # Reset loc_avg to new signal and set n_in_chop to one
            loc_avg = copy.deepcopy(output.get("signal")[i,:])
            #loc_avg = output.get("signal")[i,:]
            n_in_chop = 1
            n_eff += 1
        
            pos_old = pos_now
            nod_old = nod_now

    tot_avg = diff_container / n_nod_tot_ch

    return tot_avg

def avgDirectSubtract_chop(output):
    """!
    Calculate a new reduced timestream, averaged over chopping ON-OFF pairs.
    
    Atmosphere removal is done by direct subtraction, using the ON-OFF or ABBA scheme.
    Both schemes will work using this single method.
    Additionally, the reduced Azimuth and Elevation are returned. These are averaged over the chopping pairs

    @param output Output object obtained from a simulation.

    @returns red_signal Reduced signal.
    @returns red_Az Reduced Azimuth array.
    @returns red_El Reduced Elevation array.
    """
   
    # Determine total length of each timestream to not go out of index
    n_samps = output.get("signal").shape[0]

    # We define two counters: 
    #   - pos_old
    #   - pos_now
    # The first keeps track of the chop position of the previous timestamp.
    # The last of the current timestamp. 
    # This is done to determine in which local average the spectrum goes.
    pos_old = 0
    pos_now = 0

    # Container for storing the local average.
    # Local, as in, for storing the average in a single chop position.
    loc_avg = np.zeros(output.get("signal").shape[1])
    
    # Two containers, each for the stored average in a chopping path during an A or B.
    # When pos_now changes w.r.t. pos_old, the local average container is averaged over the total timepoints.
    # The result is either stored in ON_container (0 and 2) or OFF_container (1 and -1).
    ON_container = np.zeros(output.get("signal").shape[1])
    OFF_container = np.zeros(output.get("signal").shape[1])

    n_ON = 0
    n_OFF = 0
    # Counter for how many points to average over when averaging inside a single chop position.
    n_in_chop = 0

    # Make lists for storing timestream, azimuth and elevations
    red_signal = []
    red_Az = []
    red_El = []

    temp_Az = 0
    temp_El = 0

    for i in range(n_samps):
        # Start by updating current position
        pos_now = output.get("flag")[i]
        
        # Just keep storing in loc_avg
        if pos_now == pos_old:
            loc_avg += output.get("signal")[i,:]

            if ((pos_now == 0) or (pos_now == 2)):
                temp_Az += output.get("Az")[i]
                temp_El += output.get("El")[i]

            n_in_chop += 1

        # Encountered new chop position.
        else:
            
            # Calculate average over loc_avg in previous chop and store in appropriate container.
            # Also update approriate counter
            if ((pos_old == 0) or (pos_old == 2)):
                ON_container += loc_avg / n_in_chop
                n_ON += 1
            
            elif ((pos_old == 1) or (pos_old == -1)): 
                OFF_container += loc_avg / n_in_chop
                n_OFF += 1
            

            # Check if, by updating the numbers, a pair of ON-OFF averages has been obtained
            if n_ON == n_OFF:
                red_signal.append(ON_container - OFF_container)
                red_Az.append(temp_Az / n_in_chop)
                red_El.append(temp_El / n_in_chop)
                ON_container.fill(0)
                OFF_container.fill(0)

            # Reset loc_avg to new signal and set n_in_chop to one
            loc_avg = output.get("signal")[i,:]
            
            if ((pos_now == 0) or (pos_now == 2)):
                temp_Az = output.get("Az")[i]
                temp_El = output.get("El")[i]
            
            n_in_chop = 1
        
            pos_old = pos_now

    red_signal = np.array(red_signal)
    red_Az = np.array(red_Az)
    red_El = np.array(red_El)

    return red_signal, red_Az, red_El
