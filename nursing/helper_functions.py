# -*- coding: utf-8 -*-


# Generate the set of shifts of lengths in Lengths, starting
def GenShifts(Lengths, H, L):  # every L minutes for H hours
    assert 60 % L == 0, "L must be a divisor of 60 ..."

    # Generate the set of possible start times for shifts
    StartTimes = [60*t + l*L for t in range(H) for l in range(round(60 / L))
                  if 60*t + l*L + 60*min(Lengths) <= 60*H + 0.001]
    Shifts = []
    for l in Lengths:
        for start in StartTimes:

            # Add the possible shifts of each length
            if start + 60*l <= 60*H + 0.001:
                Shifts.append([start, start + 60*l])

            else:  # If a shift of this length doesn't fit
                continue  # now then it wont fit any later

    # Return
    return Shifts


# Convert HH:MM:SS to minutes and subtract "start"
def minutes(start, time):  # minutes from the total
    t = time.split(':')
    return (int(t[0]) * 60 + int(t[1]) + int(t[2]) / 60) - start


# Generate the matrix A used to link shift types
def GenMatrix(Shifts, L, T):  # to care worker levels

    # Most entries are zero
    A = [[0 for shift in Shifts] for t in T]
    for t in T:
        for (g, shift) in enumerate(Shifts):

            # Change A[t][shift] to 1 if shift type "shift"
            ttick = t * L + 0.01  # overlaps time period t
            if shift[0] <= ttick and ttick <= shift[1]:
                A[t][g] = 1

    # Return
    return A

