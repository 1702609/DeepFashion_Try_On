import numpy as np

if __name__ == "__main__":
    with open('area_vertical.npy', 'rb') as f:
        area_vertical = np.load(f)

    split_index = []
    single_spot = []
    index = 0
    first_item = True
    for i in area_vertical:
        if first_item:
            print("Entering the first item")
            single_spot.append(i)
            first_item = False
        else:
            if (i - 1) == single_spot[index - 1]:  # if pixels are adjacent
                single_spot.append(i)
            else:  # New array for the new spot
                split_index.append(single_spot.copy())
                single_spot.clear()
                first_item = True
                index = -1
        index += 1
    split_index.append(single_spot)