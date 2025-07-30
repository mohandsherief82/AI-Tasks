import time


def display_gear(gear_number):
    '''
    - Display a 7-segment like number based on a given number.
    - Param gear_number: represents the number of gear to display
    - Return: None
    '''
    # Initial Gear look
    gear_list = [
                 ["#", "#", "#", "#"],
                 ["#", "#", "#", "#"],
                 ["#", "#", "#", "#"],
                 ["#", "#", "#", "#"],
                 ["#", "#", "#", "#"]
                 ]

    # List representing a dictionary co-ordinates of omitted # from gear_list
    # Keys represent the number to output
    # Each value represents a dictionary with keys as row and values
    # as a list of columns giving a row and column to omit.
    segment_coordinates = {
        0: {1: [1, 2], 2: [1, 2], 3: [1, 2]},
        1: {0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2], 4: [0, 1, 2]},
        2: {1: [0, 1, 2], 3: [1, 2, 3]},
        3: {1: [0, 1, 2], 3: [0, 1, 2]},
        4: {0: [1, 2], 1: [1, 2], 3: [0, 1, 2], 4: [0, 1, 2]},
        5: {1: [1, 2, 3], 3: [0, 1, 2]},
        6: {1: [1, 2, 3], 3: [1, 2]},
        7: {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2], 4: [0, 1, 2]},
        8: {1: [1, 2], 3: [1, 2]}
    }

    # Chooses which co-ordinates from segment_coordinates based on gear_number
    output_segment = segment_coordinates[gear_number]

    # Loop to get each row of gear_list
    for row in range(5):
        if row in output_segment.keys():
            # Gets the list of omitted columns
            omit_coordinates = output_segment[row]

            # Loop changes the omitted positions from # to " ".
            for column in omit_coordinates:
                gear_list[row][column] = " "

        # Printing loop
        for column in range(4):
            print(gear_list[row][column], end="")
        print()


def gear_shift(from_gear, to_gear):
    # Uses function Display_gear to output the initial gear value
    display_gear(from_gear)

    # Pauses the change
    time.sleep(0.5)
    print("  |")
    print("  |")
    print("  |")
    print("  V")

    # Uses function Display_gear to output the final gear value
    display_gear(to_gear)


def main():

    # Stores the previous gear value
    prev_gear = 0

    # Handles all User input invalid and error inputs
    while True:
        gear_number = input("Enter Gear (0-8):")

        # Exit status to end program
        if gear_number.lower() == "exit":
            break

        # Handle string input
        try:
            # Get user input and turn it to an integer
            gear_number = int(gear_number)
        except ValueError:
            print("\nInvalid String Input!!")
            print("--------------------")
            continue

        # Handle out of range values with detailed feedback
        if gear_number > 8 or gear_number < 0:
            print("\nrInvalid number Input!!")
            print("Value must be between 0 and 8 (inclusive)!!")
            print("--------------------")
            continue
        else:
            # Display the gear grid
            gear_shift(prev_gear, gear_number)

            # Sets Prev_gear to the new value for next iteration
            prev_gear = gear_number


            print("Enter exit to stop!!")
            print("--------------------")

if __name__=="__main__":
    main()